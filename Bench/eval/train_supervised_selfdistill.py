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

# Import the self-distillation pretrainer and the classifier used for supervised training.
from train_self_distillation import VisionSelfDistillationPretrainer
from train_brats_classifier import VisionMultiLabelClassifier

#############################################
# 1. Setup Logger
#############################################
def setup_logger(log_file="train_supervised_selfdistill.log", log_to_console=True):
    logger = logging.getLogger("train_supervised_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
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
    # These arguments are used for both self-distillation and supervised training.
    model_name_or_path: str = field(
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        metadata={"help": "Path or name of the checkpoint with the vision tower."}
    )
    model_type: str = field(
        default="phi3",
        metadata={"help": "Model type to load. Options: ['llama2', 'phi3']"}
    )
    vision_tower: str = field(
        default="vit3d",
        metadata={"help": "Type of vision tower (e.g., 'vit3d')."}
    )
    pretrain_vision_model: str = field(
        default="/local2/amvepa91/M3D/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin",
        metadata={"help": "Path to pretrained ViT model."}
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze the vision tower during training."}
    )
    # New field: checkpoint from self-distillation pretraining.
    self_distill_checkpoint: str = field(
        default="./DINO_self_distillation_output/model_epoch_6.pt",
        metadata={"help": "Path to the self-distillation checkpoint (.pt file)."}
    )
    num_labels: int = field(
        default=4,
        metadata={"help": "Number of labels for multi-label classification."}
    )
    batch_size: int = field(default=2)
    num_epochs: int = field(default=50)
    learning_rate: float = field(default=1e-4)
    output_dir: str = field(default="./Supervised_SelfDistill_Output")
    device: str = field(default="cuda")
    # These parameters were used during self-distillation pretraining:
    momentum: float = field(default=0.996)
    teacher_temp: float = field(default=0.07)
    student_temp: float = field(default=0.1)
    out_dim: int = field(default=256)

#############################################
# 3. Dataset (same as before)
#############################################
class MultiLabelVisionDataset(Dataset):
    """
    Reads a JSON file, groups volumes, applies MONAI transforms,
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
        # Use appropriate transforms.
        if mode == "train":
            self.transform = mtf.Compose([
                mtf.RandRotate90(prob=0.5, spatial_axes=(1,2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])
        else:
            self.transform = mtf.Compose([
                mtf.ToTensor(dtype=torch.float),
            ])
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
            if entry.get("content_type", "") == "area":
                label = entry.get("label_name", "").strip().lower()
                answer_str = str(entry.get("answer", "None")).strip().lower()
                if label in self.label2id and answer_str != "none":
                    idx = self.label2id[label]
                    grouped[vol_dir]["label_vec"][idx] = 1.0
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
# 4. Supervised Training Loop
#############################################
def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()
    logger = setup_logger(log_file="train_supervised_selfdistill.log", log_to_console=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Replace these JSON paths with your train/val/test paths.
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_v2.json"
    val_file   = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_v2.json"
    test_file  = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json"
    
    # --------------------------------------------------------------------
    # 1) Load the pre-trained self-distillation checkpoint.
    # --------------------------------------------------------------------
    # First load the base model to get the vision tower.
    if 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    vision_tower = base_model.get_model().get_vision_tower()
    
    # Instantiate the self-distillation pretrainer.
    sd_model = VisionSelfDistillationPretrainer(vision_tower, feature_dim=2048, out_dim=args.out_dim).to(device)
    sd_model.load_state_dict(torch.load(args.self_distill_checkpoint, map_location=device))
    sd_model.eval()
    logger.info(f"Loaded self-distillation model from {args.self_distill_checkpoint}")
    
    # For supervised training, we use the student branch.
    backbone = sd_model.student
    # Ensure that only the [CLS] token is used.
    backbone.select_feature = "cls_patch"
    
    # --------------------------------------------------------------------
    # 2) Build the Multi-Label Classifier using the pre-trained backbone.
    # --------------------------------------------------------------------
    classifier = VisionMultiLabelClassifier(vision_tower=backbone, num_labels=args.num_labels).to(device)
    
    # Define optimizer for the classifier (or the entire model if fine-tuning).
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.learning_rate)
    
    # Optionally, you can define a loss function (here we use BCEWithLogitsLoss).
    criterion = nn.BCEWithLogitsLoss()
    
    # --------------------------------------------------------------------
    # 3) Create datasets and dataloaders.
    # --------------------------------------------------------------------
    train_dataset = MultiLabelVisionDataset(data_file=train_file, mode="train")
    val_dataset   = MultiLabelVisionDataset(data_file=val_file, mode="validation")
    test_dataset  = MultiLabelVisionDataset(data_file=test_file, mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    logger.info(f"Dataset sizes => train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    
    # --------------------------------------------------------------------
    # 4) Supervised Training Loop
    # --------------------------------------------------------------------
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_supervised_model.pt")
    logger.info(f"Starting supervised training for {args.num_epochs} epochs, LR={args.learning_rate}")
    
    for epoch in range(args.num_epochs):
        classifier.train()
        total_loss = 0.0
        
        for sample in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mod1 = sample["t1c"].to(device)
            mod2 = sample["t1n"].to(device)
            mod3 = sample["t2f"].to(device)
            mod4 = sample["t2w"].to(device)
            labels = sample["labels"].to(device)
            
            optimizer.zero_grad()
            # Forward pass through classifier.
            loss, logits = classifier(mod1, mod2, mod3, mod4, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = 0.0
        classifier.eval()
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mod1 = sample["t1c"].to(device)
                mod2 = sample["t1n"].to(device)
                mod3 = sample["t2f"].to(device)
                mod4 = sample["t2w"].to(device)
                labels = sample["labels"].to(device)
                batch_loss, logits = classifier(mod1, mod2, mod3, mod4, labels=labels)
                val_loss += batch_loss.item()
        val_loss /= len(val_loader)
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] - Validation Loss: {val_loss:.4f}")
        
        # Save best model checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), best_model_path)
            logger.info(f"New best validation loss: {val_loss:.4f}. Saved model to {best_model_path}")
    
    logger.info("Supervised training complete.")
    
    # --------------------------------------------------------------------
    # 5) Evaluate the Best Supervised Model on the Test Set.
    # --------------------------------------------------------------------
    logger.info("Evaluating best supervised model on test set...")
    classifier.load_state_dict(torch.load(best_model_path, map_location=device))
    classifier.eval()
    
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Test Eval"):
            mod1 = sample["t1c"].to(device)
            mod2 = sample["t1n"].to(device)
            mod3 = sample["t2f"].to(device)
            mod4 = sample["t2w"].to(device)
            labels = sample["labels"].to(device)
            logits = classifier(mod1, mod2, mod3, mod4, labels=None)
            all_labels.append(labels)
            all_logits.append(logits)
    
    all_labels = torch.cat(all_labels, dim=0)  # shape [N, num_labels]
    all_logits = torch.cat(all_logits, dim=0)  # shape [N, num_labels]
    
    all_labels_np = all_labels.cpu().numpy()
    all_probs_np = torch.sigmoid(all_logits).cpu().numpy()
    
    num_labels = all_labels_np.shape[1]
    label_aucs = []
    label_accs = []
    label_prevs = []
    from sklearn.metrics import roc_auc_score, accuracy_score
    for i in range(num_labels):
        unique_vals = np.unique(all_labels_np[:, i])
        if len(unique_vals) == 2:
            auc_i = roc_auc_score(all_labels_np[:, i], all_probs_np[:, i])
            label_aucs.append(auc_i)
        else:
            label_aucs.append(float('nan'))
        preds_binary = (all_probs_np[:, i] >= 0.5).astype(int)
        acc_i = accuracy_score(all_labels_np[:, i], preds_binary)
        label_accs.append(acc_i)
        prevalence_i = np.mean(all_labels_np[:, i])
        label_prevs.append(prevalence_i)
    
    macro_auc = np.nanmean(label_aucs)
    macro_acc = np.mean(label_accs)
    
    label_names = [
        "Non-Enhancing Tumor",
        "Surrounding Non-enhancing FLAIR hyperintensity",
        "Enhancing Tissue",
        "Resection Cavity",
    ]
    
    logger.info("========== TEST METRICS ==========")
    for i in range(num_labels):
        auc_str = f"{label_aucs[i]:.4f}" if not np.isnan(label_aucs[i]) else "N/A"
        logger.info(f"Label {i}: '{label_names[i]}' => Prevalence={label_prevs[i]*100:.2f}% | AUC={auc_str} | ACC={label_accs[i]:.4f}")
    logger.info(f"Test Macro AUC = {macro_auc:.4f}")
    logger.info(f"Test Macro Accuracy = {macro_acc:.4f}")
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
