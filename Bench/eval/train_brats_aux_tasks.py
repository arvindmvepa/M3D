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

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field


@dataclass
class VisionTrainingArguments:
    """
    Minimal training arguments for the vision classifier.
    """
    model_name_or_path: str = field(
        default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf",
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
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
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
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    output_dir: str = "./aux_classifier_output"
    device: str = "cuda"

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

# -------------------------------------------------------------------------
# 2) CORAL Utilities
# -------------------------------------------------------------------------
def coral_loss(logits, targets, K):
    """
    CORAL loss for ordinal classification with K discrete levels (0..K-1).
    logits: [N, (K-1)]  raw (no sigmoid)
    targets: [N]        integer in [0..K-1]
    We interpret logits[i] as "is class >= i+1?"
    Then do BCEWithLogits comparing to a 0/1 matrix:
      T[n,i] = 1 if targets[n] >= i+1 else 0
    """
    device = logits.device
    N = targets.size(0)
    # Build target matrix T => shape [N,(K-1)]
    T = torch.zeros((N, K - 1), device=device, dtype=torch.float)
    for i in range(K - 1):
        T[:, i] = (targets >= (i + 1)).float()

    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits, T)
    return loss

def coral_predict(logits, K):
    """
    Convert CORAL logits => integer class in [0..K-1].
    logits: [N, (K-1)]
    Return predicted class by summing how many thresholds are "passed" (>=0.5).
    """
    probs = torch.sigmoid(logits)  # => [N, (K-1)]
    passed = (probs >= 0.5).sum(dim=1)  # how many thresholds were exceeded
    return passed

# -------------------------------------------------------------------------
# 3) Soft Jaccard Loss for Bounding Boxes
# -------------------------------------------------------------------------
def soft_jaccard_loss(bbox_logits, bbox_targets, eps=1e-7):
    """
    bbox_logits: [B,4,Q]  raw
    bbox_targets: [B,4,Q] 0/1
    Returns a scalar = 1 - mean_jaccard.
    """
    p = torch.sigmoid(bbox_logits)  # => [B,4,Q]
    intersection = (p * bbox_targets).sum(dim=2)  # [B,4]
    union = (p + bbox_targets - p*bbox_targets).sum(dim=2)  # [B,4]
    jaccard = intersection / (union + eps)  # [B,4]
    return 1.0 - jaccard.mean()

# -------------------------------------------------------------------------
# 4) Multi-Task Loss: CORAL for area/extent/solidity + Soft Jaccard for bbox
# -------------------------------------------------------------------------
def compute_aux_loss_coral(
    area_logits, extent_logits, solidity_logits, bbox_logits,
    area_targets, extent_targets, solidity_targets, bbox_targets,
    K_area=10, K_extent=6, K_solidity=4
):
    """
    area_logits: [B,4,(K_area-1)]
    extent_logits: [B,4,(K_extent-1)]
    solidity_logits: [B,4,(K_solidity-1)]
    bbox_logits: [B,4,Q]
    area_targets: [B,4] in [0..K_area-1]
    extent_targets: [B,4] in [0..K_extent-1]
    solidity_targets: [B,4] in [0..K_solidity-1]
    bbox_targets: [B,4,Q] in {0,1}
    """
    B = area_logits.size(0)

    # Flatten area => [B*4,(K_area-1)], area_targets => [B*4]
    area_2d = area_logits.view(B*4, (K_area-1))
    area_tgt_1d = area_targets.view(B*4)
    area_loss = coral_loss(area_2d, area_tgt_1d, K_area)

    # extent => [B*4,(K_extent-1)]
    extent_2d = extent_logits.view(B*4, (K_extent-1))
    extent_tgt_1d = extent_targets.view(B*4)
    extent_loss = coral_loss(extent_2d, extent_tgt_1d, K_extent)

    # solidity => [B*4,(K_solidity-1)]
    solidity_2d = solidity_logits.view(B*4, (K_solidity-1))
    solidity_tgt_1d = solidity_targets.view(B*4)
    solidity_loss = coral_loss(solidity_2d, solidity_tgt_1d, K_solidity)

    # bbox => soft Jaccard
    bbox_loss = soft_jaccard_loss(bbox_logits, bbox_targets)

    total_loss = area_loss + extent_loss + solidity_loss + bbox_loss
    loss_dict = {
        "area_loss": area_loss.item(),
        "extent_loss": extent_loss.item(),
        "solidity_loss": solidity_loss.item(),
        "bbox_loss": bbox_loss.item()
    }
    return total_loss, loss_dict

# -------------------------------------------------------------------------
# 5) Dataset: CORAL + BBox
# -------------------------------------------------------------------------
class AuxVisionDataset(Dataset):
    """
    This dataset loads:
      - "area" in [0..9] => 10 ordinal levels => 9 CORAL logits
      - "extent" in [0..5] => 6 ordinal levels => 5 CORAL logits
      - "solidity" in [0..3] => 4 ordinal levels => 3 CORAL logits
      - "bbox" => multi-hot
    For 4 labels: Non-Enh, FLAIR, Enh, Resection
    We'll store them in a big dictionary: seg_file -> { label_name -> {area, extent, solidity, bbox} }
    """
    def __init__(self, json_path, mode="train", transform=None, num_quadrants=27):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.num_quadrants = num_quadrants

        self.labels_order = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]
        # Load dictionary
        with open(json_path, "r") as f:
            self.data_dict = json.load(f)

        self.samples = []
        for seg_file, label_info in self.data_dict.items():
            self.samples.append({
                "seg_file": seg_file,
                "label_info": label_info
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
        seg_file = data["seg_file"]
        label_info = data["label_info"]

        # load or create 4 volumes => [1,D,H,W] each
        # (placeholder code - adapt to real file paths)
        mod_t1c = self._random_volume()
        mod_t1n = self._random_volume()
        mod_t2f = self._random_volume()
        mod_t2w = self._random_volume()

        if self.transform is not None:
            mod_t1c = self.transform(mod_t1c)
            mod_t1n = self.transform(mod_t1n)
            mod_t2f = self.transform(mod_t2f)
            mod_t2w = self.transform(mod_t2w)

        # Prepare area, extent, solidity, bbox
        # area, extent, solidity => [4], each label an integer in correct range
        area_vals = torch.zeros(4, dtype=torch.long)
        extent_vals = torch.zeros(4, dtype=torch.long)
        solidity_vals = torch.zeros(4, dtype=torch.long)
        bbox_vals = torch.zeros((4, self.num_quadrants), dtype=torch.float)

        for i, lbl in enumerate(self.labels_order):
            if lbl in label_info:
                metrics = label_info[lbl]
                area_vals[i] = metrics["area"]         # 0..9
                extent_vals[i] = metrics["extent"]     # 0..5
                solidity_vals[i] = metrics["solidity"] # 0..3
                quad_list = metrics["bbox"]            # e.g. [16,17,20]
                for q in quad_list:
                    if q < self.num_quadrants:
                        bbox_vals[i,q] = 1.0

        return {
            "t1c": mod_t1c,
            "t1n": mod_t1n,
            "t2f": mod_t2f,
            "t2w": mod_t2w,
            "area_targets": area_vals,         # [4]
            "extent_targets": extent_vals,     # [4]
            "solidity_targets": solidity_vals, # [4]
            "bbox_targets": bbox_vals,         # [4,num_quadrants]
            "seg_file": seg_file
        }

    def _random_volume(self):
        """
        Placeholder for actual volume loading.
        Returns np array shape [1,32,256,256].
        """
        arr = np.random.randn(1, 32, 256, 256).astype(np.float32)
        return arr

# -------------------------------------------------------------------------
# 6) Model with CORAL heads for area/extent/solidity, plus a bounding-box head
# -------------------------------------------------------------------------
class VisionAuxClassifierCORAL(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        num_modalities=4,
        area_levels=10,    # => produce (area_levels-1) logits
        extent_levels=6,   # => produce (extent_levels-1) logits
        solidity_levels=4, # => produce (solidity_levels-1) logits
        num_quadrants=27   # for bbox multi-hot
    ):
        super().__init__()
        self.vision_tower = vision_tower
        hidden_dim = 768 * num_modalities  # example dimension

        # area => [B,4,(area_levels-1)]
        self.area_head = nn.Linear(hidden_dim, 4 * (area_levels - 1))

        # extent => [B,4,(extent_levels-1)]
        self.extent_head = nn.Linear(hidden_dim, 4 * (extent_levels - 1))

        # solidity => [B,4,(solidity_levels-1)]
        self.solidity_head = nn.Linear(hidden_dim, 4 * (solidity_levels - 1))

        # bbox => [B,4,num_quadrants]
        self.bbox_head = nn.Linear(hidden_dim, 4 * num_quadrants)

        self.area_levels = area_levels
        self.extent_levels = extent_levels
        self.solidity_levels = solidity_levels
        self.num_quadrants = num_quadrants

    def forward(self, mod1, mod2, mod3, mod4):
        B = mod1.size(0)

        # Extract features from each modality
        feats1 = self.vision_tower.forward(mod1)
        feats2 = self.vision_tower.forward(mod2)
        feats3 = self.vision_tower.forward(mod3)
        feats4 = self.vision_tower.forward(mod4)
        feats = torch.cat([feats1, feats2, feats3, feats4], dim=1)  # [B, 768*4]

        # area
        area_raw = self.area_head(feats)  # [B,4*(K-1)]
        area_logits = area_raw.view(B, 4, (self.area_levels - 1))

        # extent
        extent_raw = self.extent_head(feats)
        extent_logits = extent_raw.view(B, 4, (self.extent_levels - 1))

        # solidity
        solidity_raw = self.solidity_head(feats)
        solidity_logits = solidity_raw.view(B, 4, (self.solidity_levels - 1))

        # bbox
        bbox_raw = self.bbox_head(feats)  # [B,4*num_quadrants]
        bbox_logits = bbox_raw.view(B, 4, self.num_quadrants)

        return area_logits, extent_logits, solidity_logits, bbox_logits


# -------------------------------------------------------------------------
# 7) VisionTrainingArguments (same style as your previous code)
# -------------------------------------------------------------------------
@dataclass
class VisionTrainingArguments:
    model_name_or_path: str = field(
        default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf",
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
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=True, metadata={"help": "Whether to freeze vision tower weights."})

    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    output_dir: str = "./vision_corall_output"
    device: str = "cuda"


# -------------------------------------------------------------------------
# 8) The Main Training/Evaluation Script
# -------------------------------------------------------------------------
#   Adapted from your code with CORAL + Jaccard integrated
# -------------------------------------------------------------------------
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM

def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logger = setup_logger(
        log_file=f"training.log",
        log_to_console=True
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Example JSON paths for training
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_aux_v6_seed0.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_aux_v6_seed0.json"
    test_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_aux_v6_seed0.json"

    # -----------------------------------------------------------
    # 1) Load the base MLLM with a vision tower
    # -----------------------------------------------------------
    if 'llama' in args.model_type.lower():
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    elif 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}.")

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

    # Build the multi-task model
    model = VisionAuxClassifierCORAL(
        vision_tower=vision_tower,
        num_modalities=4,
        area_levels=10,      # e.g. 10 ordinal categories
        extent_levels=6,     # e.g. 6 ordinal categories
        solidity_levels=4,   # e.g. 4 ordinal categories
        num_quadrants=27     # e.g. 3x3x3 bounding box
    ).to(device)

    # -----------------------------------------------------------
    # 2) Build Datasets / Dataloaders
    # -----------------------------------------------------------
    train_dataset = AuxVisionDataset(train_file, mode="train", num_quadrants=27)
    val_dataset = AuxVisionDataset(val_file,   mode="val",   num_quadrants=27)
    test_dataset = AuxVisionDataset(test_file,  mode="test",  num_quadrants=27)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------
    # 3) Optimizer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_coral_model.pt")

    # -----------------------------------------------------------
    # 4) Training Loop
    # -----------------------------------------------------------
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            extent_targets = batch["extent_targets"].to(device)     # [B,4]
            solidity_targets = batch["solidity_targets"].to(device) # [B,4]
            bbox_targets = batch["bbox_targets"].to(device)         # [B,4,Q]

            optimizer.zero_grad()
            area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)

            loss, loss_dict = compute_aux_loss_coral(
                area_logits, extent_logits, solidity_logits, bbox_logits,
                area_targets, extent_targets, solidity_targets, bbox_targets,
                K_area=10, K_extent=6, K_solidity=4
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mod1 = batch["t1c"].to(device)
                mod2 = batch["t1n"].to(device)
                mod3 = batch["t2f"].to(device)
                mod4 = batch["t2w"].to(device)

                area_targets = batch["area_targets"].to(device)
                extent_targets = batch["extent_targets"].to(device)
                solidity_targets = batch["solidity_targets"].to(device)
                bbox_targets = batch["bbox_targets"].to(device)

                area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)
                loss, loss_dict = compute_aux_loss_coral(
                    area_logits, extent_logits, solidity_logits, bbox_logits,
                    area_targets, extent_targets, solidity_targets, bbox_targets,
                    K_area=10, K_extent=6, K_solidity=4
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best val loss = {val_loss:.4f}. Saved model to {best_model_path}")

    logger.info("Training complete.")

    # -----------------------------------------------------------
    # 5) Test Evaluation
    # -----------------------------------------------------------
    logger.info("Evaluating on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # We'll compute ordinal metrics for area, extent, solidity (e.g. MAE)
    # We'll do IoU for bbox

    all_area_preds = []
    all_area_tgts  = []
    all_extent_preds = []
    all_extent_tgts  = []
    all_solidity_preds = []
    all_solidity_tgts  = []
    iou_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            extent_targets = batch["extent_targets"].to(device)     # [B,4]
            solidity_targets = batch["solidity_targets"].to(device) # [B,4]
            bbox_targets = batch["bbox_targets"].to(device)         # [B,4,Q]

            area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)

            # 1) Area CORAL => [B,4,9]
            B = area_logits.size(0)
            area_2d = area_logits.view(B*4, 9)              # => [B*4,9]
            area_pred_1d = coral_predict(area_2d, K=10)     # => [B*4]
            area_tgt_1d = area_targets.view(-1)             # => [B*4]
            all_area_preds.append(area_pred_1d.cpu())
            all_area_tgts.append(area_tgt_1d.cpu())

            # 2) Extent => [B,4,5]
            extent_2d = extent_logits.view(B*4, 5)
            extent_pred_1d = coral_predict(extent_2d, K=6)  # => [B*4]
            extent_tgt_1d = extent_targets.view(-1)
            all_extent_preds.append(extent_pred_1d.cpu())
            all_extent_tgts.append(extent_tgt_1d.cpu())

            # 3) Solidity => [B,4,3]
            solidity_2d = solidity_logits.view(B*4, 3)
            solidity_pred_1d = coral_predict(solidity_2d, K=4) # => [B*4]
            solidity_tgt_1d = solidity_targets.view(-1)
            all_solidity_preds.append(solidity_pred_1d.cpu())
            all_solidity_tgts.append(solidity_tgt_1d.cpu())

            # 4) BBox => IoU
            bbox_prob = torch.sigmoid(bbox_logits)  # [B,4,Q]
            intersection = (bbox_prob * bbox_targets).sum(dim=2)
            union = (bbox_prob + bbox_targets - bbox_prob*bbox_targets).sum(dim=2)
            iou = intersection / (union + 1e-7)   # [B,4]
            iou_list.append(iou.cpu())

    # stack predictions
    area_preds = torch.cat(all_area_preds).numpy()
    area_tgts  = torch.cat(all_area_tgts).numpy()
    extent_preds = torch.cat(all_extent_preds).numpy()
    extent_tgts  = torch.cat(all_extent_tgts).numpy()
    solidity_preds = torch.cat(all_solidity_preds).numpy()
    solidity_tgts  = torch.cat(all_solidity_tgts).numpy()
    iou_tensor = torch.cat(iou_list, dim=0) # shape [N*B, 4]
    mean_iou = iou_tensor.mean().item()

    # Simple metrics: Mean Absolute Error for ordinal
    area_mae = mean_absolute_error(area_tgts, area_preds)
    extent_mae = mean_absolute_error(extent_tgts, extent_preds)
    solidity_mae = mean_absolute_error(solidity_tgts, solidity_preds)

    logger.info("========== TEST RESULTS ==========")
    logger.info(f"Area MAE:     {area_mae:.4f}")
    logger.info(f"Extent MAE:   {extent_mae:.4f}")
    logger.info(f"Solidity MAE: {solidity_mae:.4f}")
    logger.info(f"BBox Mean IoU:{mean_iou:.4f}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
