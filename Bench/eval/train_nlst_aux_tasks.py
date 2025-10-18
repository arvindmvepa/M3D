import os
import sys
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import re
import collections

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from dataclasses import dataclass, field
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM


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


_RULES = [
    (0, re.compile(r"\bb7\d|b70f", re.I)),  # Siemens very-sharp
    (1, re.compile(r"\bb50f?", re.I)),  # Siemens sharp
    (1, re.compile(r"\bbone|lspluslung|qxd|lung", re.I)),  # GE bone / lung
    (1, re.compile(r"fc5\d", re.I)),  # Toshiba FC51/FC53
    (1, re.compile(r"\bphil.*d\b", re.I)),  # Philips D kernels
    (2, re.compile(r"\bb40f", re.I)),  # Siemens medium-sharp
    (3, re.compile(r"\bb3\d+f?", re.I)),  # Siemens B30 family
    (3, re.compile(r"\bfc10|fc0[12]", re.I)),  # Toshiba FC10/FC02/FC01
    (4, re.compile(r"\bstandard|std", re.I)),  # GE Standard
    (4, re.compile(r"\bphil.*[bc]\b", re.I)),  # Philips C / B
    (9, re.compile(r".*")),  # fallback: worst
]


def get_npy_path(volume_path, img_root="/data/lung/nlst/NLST_CT_npy"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


def bce_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())


@torch.no_grad()
def eval_baseline(loader, majority, mean_val, device):
    agg, cnt = collections.defaultdict(float), collections.defaultdict(int)

    for batch in loader:
        for k, tgt in batch["label_dict"].items():         # tgt: [B, L]
            tgt = tgt.to(device)
            if "diameter" in k:
                pred = mean_val[k].to(device).unsqueeze(0).expand_as(tgt)  # [B,L]
                mask = torch.isfinite(tgt)
                agg[k] += ((pred - tgt)[mask] ** 2).sum().item()
                cnt[k] += mask.sum().item()
            else:
                pred = majority[k].to(device).unsqueeze(0).expand_as(tgt)  # [B,L]
                mask = torch.isfinite(tgt)
                agg[k] += (pred[mask] == tgt.long()[mask]).sum().item()
                cnt[k] += mask.sum().item()

    out = {}
    for k in agg:
        metric_name = k.replace("_labels", "") + ("_mse" if "diameter" in k else "_acc")
        out[metric_name] = agg[k] / max(1, cnt[k])
    return out


def compute_aux_loss(logits, targets):

    cancer_loss = bce_loss(logits.squeeze(-1), targets.squeeze(-1))
    total_loss = cancer_loss

    return total_loss


class AuxVisionDataset(Dataset):

    def __init__(self, json_path, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform

        # Load dictionary
        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "img_files": datum_dict["img_files"],
                "filters": datum_dict["filters"],
                "target": datum_dict["numeric_answer"]
            })

        # If no transform is provided, define a default
        if self.transform is None:
            if mode == "train":
                self.transform = mtf.Compose([
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        # load img information
        img_files = data["img_files"]
        filters = data["filters"]
        best_filter_index = self.best_filter_index(filters)
        img_file = img_files[best_filter_index]
        img_file_npy = get_npy_path(img_file)
        target = data["target"]

        # Load img file
        img_npy = np.load(img_file_npy)

        if self.transform is not None:
            img_tensor = self.transform(img_npy)

        return {
            "image": img_tensor,
            "target": target,
        }

    def convert_file_path_to_npy(self, image_abs_path):
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"
        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path

    def _priority(self, kernel) -> int:
        """Return an integer priority; lower = better for lung work."""
        if isinstance(kernel, str):
            for rank, pattern in _RULES:
                if pattern.search(kernel):
                    return rank
        return 9 

    def best_filter_index(self, filters) -> int:
        """
        Given a list of convolution-kernel strings, return the index
        (0-based) of the kernel most suitable for lung-nodule evaluation.
        If several share the same priority, the first in the list wins.
        """
        if not filters:
            raise ValueError("Empty filter list")

        priorities = [self._priority(k) for k in filters]
        return priorities.index(min(priorities))


class VisionAuxClassifier(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        use_cls=False
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.use_cls = use_cls
        self.cls_hidden_dim = 768
        self.non_cls_hidden_dim = 768 * 2048
        if self.use_cls:
            self.cancer_head = nn.Linear(self.cls_hidden_dim, 1)
        else:
            self.cancer_head = nn.Linear(self.non_cls_hidden_dim, 1)

    def forward(self, image):
        B = image.size(0)

        # Extract features
        feats = self.vision_tower.forward(image)

        if self.use_cls:
            # collect cls and non-cls features
            cls_feats = feats[:, 0]
            cls_feats = cls_feats.view(B, self.cls_hidden_dim)
            mdl_feats = cls_feats
        else:
            non_cls_feats = feats
            non_cls_feats = non_cls_feats.view(B, self.non_cls_hidden_dim)
            mdl_feats = non_cls_feats

        logits = self.cancer_head(mdl_feats).view(B, 1)
        return logits


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
        metadata={"help": "Which vision tower in the loaded model (e.g. 'vit3d')."}
    )
    pretrain_vision_model: str = field(default="./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin",
                                       metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=True, metadata={"help": "Whether to freeze vision tower weights."})

    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    output_dir: str = "./nlst_vision_cancer_aux_output"
    device: str = "cuda"
    tag: str = ""
    use_cls: bool = True


def evaluate(loader, model, device):
    model.eval()

    all_predictions = []
    all_targets = []
    all_logits = []

    for batch in loader:
        img = batch["image"].to(device)
        target = batch["target"].to(device)
        target = target.squeeze(-1).long() # Remove last dimension if present
        out = model(img)

        # Get logits and apply sigmoid for probabilities
        logits = out.squeeze(-1)  # Remove last dimension if present
        probs = torch.sigmoid(logits)

        # Convert to binary predictions (threshold at 0.5)
        preds = (probs > 0.5).long()

        # Collect all results
        all_logits.extend(probs.detach().cpu().numpy())
        all_predictions.extend(preds.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)

    # AUC (only if we have both classes)
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_logits)
    else:
        auc = 0.0  # Cannot compute AUC with only one class

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }


def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_use_cls_{args.use_cls}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(
        log_file=os.path.join(output_dir,
                              f"aux_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_use_cls_{args.use_cls}.log"),
        log_to_console=True
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Example JSON paths for training
    train_file = "/home/avepa/MedTrinity-25M/nlst_aux_cancer_train_v5.json"
    val_file = "/home/avepa/MedTrinity-25M/nlst_aux_cancer_val_v5.json"
    test_file = "/home/avepa/MedTrinity-25M/nlst_aux_cancer_test_v5.json"

    if 'llama' in args.model_type.lower():
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    elif 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}.")
    logger.info(f"Loaded model from {args.model_name_or_path}")

    vision_tower = base_model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError("No vision tower found in the loaded model.")

    # If we have a pretrained vision tower checkpoint
    if args.pretrain_vision_model is not None:
        state_dict = torch.load(args.pretrain_vision_model)
        updated_state_dict = {"vision_tower." + k: v for k, v in state_dict.items()}
        vision_tower.load_state_dict(updated_state_dict, strict=False)
        logger.info(f"Loaded vision tower from {args.pretrain_vision_model}")

    # Freeze if requested
    if args.freeze_vision_tower:
        for param in vision_tower.parameters():
            param.requires_grad = False
        logger.info("Vision tower is frozen.")

    if args.use_cls:
        vision_tower.select_feature = 'cls_patch'
        logger.info("Using [CLS] token during training.")

    # Build the multi-task model
    model = VisionAuxClassifier(
        vision_tower=vision_tower,
        use_cls=args.use_cls
    ).to(device)

    # -----------------------------------------------------------
    # 2) Build Datasets / Dataloaders
    # -----------------------------------------------------------
    train_dataset = AuxVisionDataset(train_file, mode="train")
    val_dataset = AuxVisionDataset(val_file, mode="val")
    test_dataset = AuxVisionDataset(test_file, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, drop_last=True)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------
    # 3) Optimizer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        comp_sums = collections.defaultdict(float)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            image = batch["image"].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(image)

            loss = compute_aux_loss(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        logger.info(f"[Epoch {epoch + 1}] Train loss = {avg_train_loss:.5f}")
        model.eval()
        val_total_loss = 0.0
        val_comp_sums = collections.defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                image = batch["image"].to(device)
                target = batch["target"].to(device)

                logits = model(image)
                v_loss = compute_aux_loss(logits, target)

                val_total_loss += v_loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        val_metrics = evaluate(val_loader, model, device)

        logger.info(f"[Epoch {epoch + 1}]  Val loss = {avg_val_loss:.5f} " +
                    " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

        # ------------- save best checkpoint ------------------- #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ➜ New best model saved ({best_val_loss:.5f})")

    # =========================================================== #
    # Test-set evaluation with the best checkpoint
    # =========================================================== #
    logger.info("==========  TEST  ==========")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            image = batch["image"].to(device)
            target = batch["target"].to(device)

            logits = model(image)
            t_loss = compute_aux_loss(logits, target)

            test_total_loss += t_loss.item()

    avg_test_loss = test_total_loss / len(test_loader)
    test_metrics = evaluate(test_loader, model, device)

    logger.info(f"Best-val model Test loss = {avg_test_loss:.5f} " +
                " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))


if __name__ == "__main__":
    main()
