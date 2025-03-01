import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import json
import logging
from tqdm import tqdm
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import copy

# Add your project path so your modules are found.
sys.path.append('/home/acc/NKG/M3D-branch/M3D')
import monai.transforms as mtf
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM

#############################################
# 1. Setup Logger
#############################################
def setup_logger(log_file="training.log", log_to_console=True):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

#############################################
# 2. Define Training Arguments
#############################################
@dataclass
class VisionTrainingArguments:
    model_name_or_path: str = field(
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        metadata={"help": "Path or name of the checkpoint that contains the vision tower."}
    )
    model_type: str = field(
        default="phi3",
        metadata={"help": "Model type to load. Options: ['llama2', 'phi3']"}
    )
    vision_tower: str = field(
        default="vit3d",
        metadata={"help": "Whether we have a vision tower in the loaded model (e.g. 'vit3d')."}
    )
    pretrain_vision_model: str = field(
        default="/local2/amvepa91/M3D/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin",
        metadata={"help": "Path to pretrained model for ViT."}
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision tower during pretraining."}
    )
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    output_dir: str = "./DINO_self_distillation_output4"
    device: str = "cuda"
    momentum: float = 0.996      # EMA momentum for teacher update
    teacher_temp: float = 0.07
    student_temp: float = 0.1
    out_dim: int = 256

#############################################
# 3. Dataset (unchanged from your code)
#############################################
class MultiLabelVisionDataset(Dataset):
    """
    Reads your JSON file, groups volumes, applies MONAI transforms,
    and returns 4 modalities per sample.
    """
    def __init__(self, data_file, mode="train"):
        super().__init__()
        self.data_file = data_file
        self.mode = mode
        self.specified_labels = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]
        self.label2id = {lbl.lower(): i for i, lbl in enumerate(self.specified_labels)}
        self.num_labels = len(self.specified_labels)
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1,2)),
            mtf.RandFlip(prob=0.1, spatial_axis=0),
            mtf.RandFlip(prob=0.1, spatial_axis=1),
            mtf.RandFlip(prob=0.1, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        self.transform = train_transform if mode=="train" else val_transform
        self.data_by_vol = self._read_and_group_data(self.data_file)
        self.samples = []
        for vol_dir, vol_info in self.data_by_vol.items():
            self.samples.append({
                "volume_file_dir": vol_dir,
                "volume_non_seg_files": vol_info["volume_non_seg_files"],
                "label_vec": vol_info["label_vec"],
            })

    def _read_and_group_data(self, json_path):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        grouped = {}
        for entry in raw_data:
            vol_dir = entry["volume_file_dir"]
            if vol_dir not in grouped:
                grouped[vol_dir] = {
                    "volume_non_seg_files": entry["volume_non_seg_files"],
                    "label_vec": torch.zeros(self.num_labels, dtype=torch.float),
                }
            if entry.get("content_type", "")=="area":
                label = entry.get("label_name", "").strip().lower()
                answer_str = str(entry.get("answer", "None")).strip().lower()
                if label in self.label2id and answer_str!="none":
                    idx = self.label2id[label]
                    grouped[vol_dir]["label_vec"][idx]=1.0
        return grouped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vol_dir = sample["volume_file_dir"]
        label_vec = sample["label_vec"].clone()
        modalities = ["t1c", "t1n", "t2f", "t2w"]
        returned_dict = {}
        for modality in modalities:
            npy_path = sample["volume_non_seg_files"][modality]
            npy_path = self.convert_file_path_to_npy(npy_path)
            vol_data = np.load(npy_path)
            vol_data = self.transform(vol_data)
            returned_dict[modality] = vol_data
        returned_dict["labels"] = label_vec
        returned_dict["volume_file_dir"] = vol_dir
        return returned_dict

    def convert_file_path_to_npy(self, image_abs_path):
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"
        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path

#############################################
# 4. Define the DINO Loss
#############################################
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        # Ensure center is on the same device as teacher_output.
        center = self.center.to(teacher_output.device)
        teacher_out = torch.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
        student_out = torch.log_softmax(student_output / self.student_temp, dim=-1)
        loss = - torch.sum(teacher_out * student_out, dim=-1).mean()
        # Compute new center from teacher_output.
        new_center = teacher_output.mean(dim=0, keepdim=True)
        # Update center in-place so that self.center stays on the same device.
        self.center.copy_(center * self.center_momentum + new_center * (1 - self.center_momentum))
        return loss

#############################################
# 5. Define the Updated Self-Distillation Model
#############################################
class VisionSelfDistillationPretrainer(nn.Module):
    def __init__(self, vision_tower: nn.Module, feature_dim: int, out_dim: int = 256):
        super().__init__()
        self.student = vision_tower
        # Create teacher as a deep copy and freeze its parameters.
        self.teacher = copy.deepcopy(vision_tower)
        for param in self.teacher.parameters():
            param.requires_grad = False
        # Projection heads map features to the desired output dimension.
        self.student_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, out_dim)
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, out_dim)
        )

    def forward(self, views):
        """
        Expects a list of augmented views (tensors).
        Assumes that the first two views are used for the teacher (global views)
        and the remaining views for the student.
        Processes each view, extracts the [CLS] token from the primary output,
        passes it through the corresponding projection head, and then aggregates
        (averages) the outputs across views.
        """
        teacher_views = views[:2]
        student_views = views[2:]
        
        teacher_outs = []
        # Compute teacher outputs under no_grad.
        with torch.no_grad():
            for x in teacher_views:
                outputs = self.teacher(x)  # outputs[0] contains features [B, num_tokens, hidden_dim]
                features = outputs[0]
                feat = features[:, 0]      # select the [CLS] token
                teacher_outs.append(self.teacher_head(feat))
            
        student_outs = []
        for x in student_views:
            outputs = self.student(x)
            features = outputs[0]
            feat = features[:, 0]
            student_outs.append(self.student_head(feat))
        
        # Aggregate (average) the outputs across views.
        teacher_agg = torch.stack(teacher_outs, dim=0).mean(dim=0)
        student_agg = torch.stack(student_outs, dim=0).mean(dim=0)
        return student_agg, teacher_agg

    @torch.no_grad()
    def update_teacher(self, momentum):
        """
        Update teacher parameters using EMA of student parameters.
        """
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = teacher_param.data * momentum + student_param.data * (1 - momentum)
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data = teacher_param.data * momentum + student_param.data * (1 - momentum)

#############################################
# 6. Updated Training Loop
#############################################
def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()
    logger = setup_logger(
        log_file=f"self_distillation_{os.path.basename(args.model_name_or_path)}.log",
        log_to_console=True
    )
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Define JSON file paths (update as needed)
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_v2.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_v2.json"
    
    # 6.1 Load the pre-trained MLLM with a vision tower.
    if 'llama' in args.model_type.lower():
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    elif 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Unknown model type.")
    
    vision_tower = base_model.get_model().get_vision_tower()
    if args.pretrain_vision_model is not None:
        state_dict = torch.load(args.pretrain_vision_model)
        updated_state_dict = {"vision_tower." + k: v for k, v in state_dict.items()}
        vision_tower.load_state_dict(updated_state_dict)
        logger.info(f"Loaded vision tower from {args.pretrain_vision_model}")
    
    if args.freeze_vision_tower:
        for param in vision_tower.parameters():
            param.requires_grad = False
        logger.info("Vision tower is frozen.")
    else:
        logger.info("Vision tower is unfrozen. Training student network.")
    
    # 6.2 Build the self-distillation pretrainer model.
    # Here we assume the vision tower outputs features with dimension 2048.
    model = VisionSelfDistillationPretrainer(vision_tower, feature_dim=2048, out_dim=args.out_dim).to(device)
    optimizer = optim.AdamW(model.student.parameters(), lr=args.learning_rate)
    
    # 6.3 Create dataset and dataloader.
    train_dataset = MultiLabelVisionDataset(data_file=train_file, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 6.4 Create the DINOLoss instance.
    dino_loss_fn = DINOLoss(out_dim=args.out_dim, teacher_temp=args.teacher_temp, student_temp=args.student_temp)
    
    logger.info(f"Starting self-distillation pretraining for {args.num_epochs} epochs.")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for sample in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            # Use the four modalities as different views.
            mod1 = sample["t1c"].to(device)
            mod2 = sample["t1n"].to(device)
            mod3 = sample["t2f"].to(device)
            mod4 = sample["t2w"].to(device)
            # Designate the first two views for the teacher and the remaining for the student.
            views = [mod1, mod2, mod3, mod4]
            
            optimizer.zero_grad()
            student_output, teacher_output = model(views)
            loss = dino_loss_fn(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            # Update teacher with EMA.
            model.update_teacher(args.momentum)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] - Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    logger.info("Self-distillation pretraining complete.")

if __name__ == "__main__":
    main()
