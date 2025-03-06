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

# Add sklearn for metrics
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import HfArgumentParser
from dataclasses import dataclass, field

import sys

sys.path.append('/local2/amvepa91/M3D')
import monai.transforms as mtf
from LaMed.src.model.loss import MultiModalContrastiveLoss, GroupContrastiveLoss
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM

from transformers import HfArgumentParser, AutoModel, AutoConfig


def setup_logger(log_file="training.log", log_to_console=True):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Optional console handler
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


@dataclass
class VisionTrainingArguments:
    """
    Minimal training arguments for the vision classifier.
    """
    model_name_or_path: str = field(
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        # default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf",
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
    pretrain_vision_model: str = field(default=
                                       # "GoodBaiBai88/M3D-CLIP",
                                       "/local2/amvepa91/M3D/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin",
                                       metadata={"help": "Path to pretrained model for ViT."})
    pretrain_mllm: str = field(
        default=None,
        metadata={"help": "Path to a pretrained MLLM weights to load into the model (optional)."}
    )

    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the entire vision tower during training."}
    )

    num_labels: int = field(
        default=4,
        metadata={"help": "Number of labels for multi-label classification."}
    )

    # Basic training settings
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    output_dir: str = "./SimCLR_vision_classifier_output_met"
    device: str = "cuda"
    train_file: str = "/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_train_v1.json"
    val_file: str = "/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_val_v1.json"
    test_file: str = "/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_test_v1.json"


class MultiLabelVisionDataset(Dataset):
    """
    A multi-label vision-only dataset that:
      1) Reads the same JSON structure used by VQABratsDataset.
      2) Groups entries by 'volume_file_dir'.
      3) Only uses *four* labels (ignoring 'Tumor Core'):
         - Non-Enhancing Tumor
         - Surrounding Non-enhancing FLAIR hyperintensity
         - Enhancing Tissue
         - Resection Cavity
      4) For each label, presence = (content_type=="area" AND answer!="None").
      5) Creates a 4D multi-hot label vector per volume.
      6) Loads 4 modalities per volume (t1c, t1n, t2f, t2w).
      7) Returns {'t1c':..., 't1n':..., 't2f':..., 't2w':..., 'labels':..., 'volume_file_dir':...}.
    """

    def __init__(self, data_file, mode="train"):
        """
        :param data_file: Path to your JSON file (train/val/test).
        :param mode:      'train', 'validation', or 'test'.
        """
        super().__init__()
        self.data_file = data_file
        self.mode = mode

        # ------------------------------------------------------
        # We define exactly four labels and ignore "Tumor Core"
        # ------------------------------------------------------
        self.specified_labels = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]
        # Make them lowercase for matching
        self.label2id = {lbl.lower(): i for i, lbl in enumerate(self.specified_labels)}
        self.num_labels = len(self.specified_labels)  # 4

        # ------------------------------------------------------
        # 2) Define transforms (similar to VQABrats code)
        # ------------------------------------------------------
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        if mode == "train":
            self.transform = train_transform
        elif mode in ["validation", "test"]:
            self.transform = val_transform
        else:
            raise ValueError(f"Unknown mode {mode}.")

        # ------------------------------------------------------
        # 3) Read JSON & group by volume_file_dir
        # ------------------------------------------------------
        self.data_by_vol = self._read_and_group_data(self.data_file)

        # ------------------------------------------------------
        # 4) Flatten grouped volumes into a list for __getitem__
        # ------------------------------------------------------
        self.samples = []
        for vol_dir, vol_info in self.data_by_vol.items():
            self.samples.append({
                "volume_file_dir": vol_dir,
                "volume_non_seg_files": vol_info["volume_non_seg_files"],
                "label_vec": vol_info["label_vec"],  # 4-dim multi-hot
            })

    def _read_and_group_data(self, json_path):
        """
        1) Load the JSON (like VQABratsDataset).
        2) Group by volume_file_dir.
        3) For each volume_file_dir:
             - volume_non_seg_files: from the *first* matching entry
             - label_vec: a 4-dim multi-hot vector
        """
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

            # We only care about content_type == "area"
            if entry.get("content_type", "") == "area":
                label = entry.get("label_name", "").strip().lower()
                answer_str = str(entry.get("answer", "None")).strip().lower()

                # If label is among our four, mark it if answer != 'none'
                if label in self.label2id and answer_str != "none":
                    idx = self.label2id[label]
                    grouped[vol_dir]["label_vec"][idx] = 1.0

        return grouped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vol_dir = sample["volume_file_dir"]
        label_vec = sample["label_vec"].clone()  # shape [4]

        # Load each modality => shape [C, D, H, W], then transform
        modalities = ["t1c", "t1n", "t2f", "t2w"]
        returned_dict = {}
        for modality in modalities:
            npy_path = sample["volume_non_seg_files"][modality]
            npy_path = self.convert_file_path_to_npy(npy_path)
            vol_data = np.load(npy_path)  # shape e.g. [1, 32, 256, 256]
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


class VisionMultiLabelClassifier(nn.Module):
    def __init__(self, vision_tower: nn.Module, num_labels: int, num_modalities=4, contrastive_temp=0.5,
                 contrastive_weight=1.0):
        """
        :param vision_tower: The extracted vision model.
        :param num_labels:   Number of labels for multi-label classification.
        :param num_modalities: How many modalities we are concatenating feature-wise.
        """
        super().__init__()
        self.vision_tower = vision_tower
        # We'll assume each single modality forward returns shape [B, 768]
        hidden_dim = 768 * num_modalities
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # self.projection_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(768, 768),
        #         nn.ReLU(),
        #         nn.Linear(768, 128)
        #     ) for _ in range(num_modalities)
        # ])

        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.contrastive_loss = MultiModalContrastiveLoss(temperature=contrastive_temp)
        self.contrastive_loss = GroupContrastiveLoss(temperature=contrastive_temp)
        self.contrastive_weight = contrastive_weight

    def forward(self, mod1, mod2, mod3, mod4, labels=None):
        # [batch_size, 1, 32, 256, 256]
        feats1 = self.vision_tower.forward(mod1)
        feats2 = self.vision_tower.forward(mod2)
        feats3 = self.vision_tower.forward(mod3)
        feats4 = self.vision_tower.forward(mod4)

        proj_feats = []
        all_feats = [feats1, feats2, feats3, feats4]
        # for idx, feats in enumerate(all_feats):
        #     proj = self.projection_heads[idx](feats)
        #     proj_feats.append(proj)

        combined_feats = torch.cat([feats1, feats2, feats3, feats4], dim=1)
        logits = self.classifier(combined_feats)

        if labels is not None:
            class_loss = self.bce_loss(logits, labels)

            # cont_loss = self.contrastive_loss(all_feats)
            cont_loss = self.contrastive_loss(all_feats, labels)

            total_loss = class_loss + self.contrastive_weight * cont_loss
            return total_loss, logits
        else:
            return logits


def main(train_file="/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_train_v1.json",
         val_file="/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_val_v1.json",
         test_file="/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_test_v1.json"):
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()
    version = "v2"
    logger = setup_logger(
        log_file=f"model_name_{os.path.basename(args.model_name_or_path)}_pretrained_vision_tower_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_{version}.log",
        log_to_console=True
    )
    output_dir = args.output_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------
    # 1) Load the pre-trained MLLM with a vision tower
    # --------------------------------------------------------------------
    if 'llama' in args.model_type.lower():
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    elif 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}. Supported: ['llama2', 'phi3']")

    print('args.model_name_or_path', args.model_name_or_path)
    print('os.path.basename(args.model_name_or_path)', os.path.basename(args.model_name_or_path))
    print('args.pretrain_vision_model', args.pretrain_vision_model)
    print('os.path.basename(args.pretrain_vision_model)', os.path.basename(args.pretrain_vision_model))

    vision_tower = base_model.get_model().get_vision_tower()
    if args.pretrain_vision_model is not None:
        if 'GoodBaiBai88/M3D-CLIP' in args.pretrain_vision_model:
            state_dict = torch.load("./m3d_clip_model/pretrained_ViT.bin")
            updated_state_dict = {"vision_tower." + k: v for k, v in state_dict.items()}
            vision_tower.load_state_dict(updated_state_dict)
        else:
            state_dict = torch.load(args.pretrain_vision_model)
        # add vision_tower to the state_dict
        updated_state_dict = {"vision_tower." + k: v for k, v in state_dict.items()}
        vision_tower.load_state_dict(updated_state_dict)
        logger.info(f"Loaded vision tower from {args.pretrain_vision_model}")
    vision_tower.select_feature = "cls_patch"
    if vision_tower is None:
        raise ValueError("No vision tower found in the loaded model. Check `vision_tower` args.")

    if args.freeze_vision_tower:
        for param in vision_tower.parameters():
            param.requires_grad = False
        logger.info("Vision tower is frozen.")
    else:
        logger.info("Vision tower is unfrozen. Fine-tuning it.")

    model = VisionMultiLabelClassifier(vision_tower=vision_tower, num_labels=args.num_labels).to(device)

    # --------------------------------------------------------------------
    # 2) Datasets / Dataloaders
    # --------------------------------------------------------------------
    train_dataset = MultiLabelVisionDataset(data_file=args.train_file, mode="train")
    val_dataset = MultiLabelVisionDataset(data_file=args.val_file, mode="validation")
    test_dataset = MultiLabelVisionDataset(data_file=args.test_file, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # --------------------------------------------------------------------
    # 3) Optimizer
    # --------------------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # --------------------------------------------------------------------
    # 4) Training Loop
    # --------------------------------------------------------------------
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training for {args.num_epochs} epochs, LR={args.learning_rate}")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for sample in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            mod1 = sample["t1c"].to(device)
            mod2 = sample["t1n"].to(device)
            mod3 = sample["t2f"].to(device)
            mod4 = sample["t2w"].to(device)
            labels = sample["labels"].to(device)

            optimizer.zero_grad()
            loss, logits = model(mod1, mod2, mod3, mod4, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # ------------------------------
        # Validation
        # ------------------------------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                mod1 = sample["t1c"].to(device)
                mod2 = sample["t1n"].to(device)
                mod3 = sample["t2f"].to(device)
                mod4 = sample["t2w"].to(device)
                labels = sample["labels"].to(device)

                batch_loss, logits = model(mod1, mod2, mod3, mod4, labels=labels)
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}] - Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best val loss = {val_loss:.4f}. Saved model to {best_model_path}")

    logger.info("Training complete.")
    # --------------------------------------------------------------------
    # 5) Evaluate best model on the test set: AUC-ROC & Accuracy & Label Prevalence
    # --------------------------------------------------------------------
    logger.info("Evaluating on the test set using best checkpoint...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Test Eval"):
            mod1 = sample["t1c"].to(device)
            mod2 = sample["t1n"].to(device)
            mod3 = sample["t2f"].to(device)
            mod4 = sample["t2w"].to(device)
            labels = sample["labels"].to(device)

            # Forward pass without computing loss
            logits = model(mod1, mod2, mod3, mod4, labels=None)
            all_labels.append(labels)
            all_logits.append(logits)

    # Stack everything
    all_labels = torch.cat(all_labels, dim=0)  # shape [N, 4]
    all_logits = torch.cat(all_logits, dim=0)  # shape [N, 4]

    # Convert to CPU numpy
    all_labels_np = all_labels.cpu().numpy()  # 0/1 for each label
    all_probs_np = torch.sigmoid(all_logits).cpu().numpy()  # [0..1] for each label

    num_labels = all_labels_np.shape[1]

    # ------------------------------------------------------------
    # Compute AUC for each label, then macro-average
    # ------------------------------------------------------------
    label_aucs = []
    for i in range(num_labels):
        unique_vals = np.unique(all_labels_np[:, i])
        if len(unique_vals) == 2:
            auc_i = roc_auc_score(all_labels_np[:, i], all_probs_np[:, i])
            label_aucs.append(auc_i)
        else:
            label_aucs.append(float('nan'))

    macro_auc = np.nanmean(label_aucs)

    # ------------------------------------------------------------
    # Compute multi-label accuracy
    #   We'll threshold each probability at 0.5, compare to ground truth.
    #   Then compute per-label accuracy, and macro-average across labels.
    # ------------------------------------------------------------
    preds_binary = (all_probs_np >= 0.5).astype(int)  # shape [N, 4]
    label_accs = []
    for i in range(num_labels):
        acc_i = accuracy_score(all_labels_np[:, i], preds_binary[:, i])
        label_accs.append(acc_i)
    macro_acc = np.mean(label_accs)

    # ------------------------------------------------------------
    # Compute label prevalence for the test set
    #   label_prevalence = fraction of test samples that have label i = 1
    # ------------------------------------------------------------
    label_prevs = []
    for i in range(num_labels):
        prevalence_i = np.mean(all_labels_np[:, i])  # fraction of 1's
        label_prevs.append(prevalence_i)

    # Log final metrics
    label_names = [
        "Non-Enhancing Tumor",
        "Surrounding Non-enhancing FLAIR hyperintensity",
        "Enhancing Tissue",
        "Resection Cavity",
    ]

    logger.info("========== TEST METRICS ==========")
    for i in range(num_labels):
        label_str = label_names[i]
        auc_str = f"{label_aucs[i]:.4f}" if not np.isnan(label_aucs[i]) else "N/A"
        logger.info(
            f"Label {i}: '{label_str}' "
            f"=> Prevalence={label_prevs[i] * 100:.2f}% | "
            f"AUC={auc_str} | "
            f"ACC={label_accs[i]:.4f}"
        )

    logger.info(f"Test Macro AUC = {macro_auc:.4f}")
    logger.info(f"Test Macro Accuracy = {macro_acc:.4f}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()