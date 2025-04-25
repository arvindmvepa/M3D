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

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import mean_absolute_error
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


def coral_predict(logits):
    probs = torch.sigmoid(logits)
    passed = (probs >= 0.5).sum(dim=1)  # how many thresholds were exceeded
    return passed


def ce_predict(logits):
    probs = torch.softmax(logits, dim=1)  # => [N, (K-1)]
    preds = torch.argmax(probs, dim=1)  # => [N]
    return preds


def mse_loss(logits, labels):
    """
    Builds a multi-hot [B, L, 27] then does MSE between
    sigmoid(logits) and the labels. Returns a scalar.
    """
    B, L, Q = logits.shape
    if Q != 27:
        raise ValueError(f"Expected 27 quadrants, got Q={Q}.")
    B_, L_, Q_ = labels.shape
    if Q_ != 27:
        raise ValueError(f"(labels) Expected 27 quadrants, got Q={Q_}.")

    # Convert logits -> probabilities
    probs = torch.sigmoid(logits)  # shape [B, L, 27]

    # Compute standard MSE
    loss_mse = F.mse_loss(probs, labels)
    return loss_mse


def bce_loss(logits, labels):
    B, L, Q = logits.shape
    if Q != 11:
        raise ValueError(f"(logits) Expected 11 regions, got Q={Q}.")
    B_, L_, Q_ = labels.shape
    if Q_ != 11:
        raise ValueError(f"(labels) Expected 11 regions, got Q={Q_}.")

    # Compute BCE with logits
    loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
    return loss_bce


def ce_loss(logits, labels):
    loss_ce = F.cross_entropy(logits, labels)
    return loss_ce


def compute_aux_loss(
    area_logits, shape_logits, satellite_logits, region_logits,
    area_targets, shape_targets, satellite_targets, region_targets,
    K_area=10, K_shape=7, K_satellite=5, keep_only_region=False,
    region_loss="bce"
):
    B = area_logits.size(0)

    area_2d = area_logits.view(B*4, (K_area-1))
    area_tgt_1d = area_targets.view(B*4)
    area_loss = coral_loss(area_2d, area_tgt_1d, K_area)

    shape_2d = shape_logits.view(B * 4, K_shape)
    shape_tgt_1d = shape_targets.view(B * 4)
    shape_loss = ce_loss(shape_2d, shape_tgt_1d)

    # solidity => [B*4,(K_solidity-1)]
    satellite_2d = satellite_logits.view(B * 4, K_satellite)
    satellite_tgt_1d = satellite_targets.view(B * 4)
    satellite_loss = ce_loss(satellite_2d, satellite_tgt_1d)

    if region_loss == "bce":
        region_loss = bce_loss(region_logits, region_targets)
    else:
        raise ValueError(f"Unknown bbox_loss: {region_loss}")

    if keep_only_region:
        total_loss = region_loss
    else:
        total_loss = area_loss + shape_loss + satellite_loss + region_loss
    loss_dict = {
        "area_loss": area_loss.item(),
        "shape_loss": shape_loss.item(),
        "satellite_loss": satellite_loss.item(),
        "region_loss": region_loss.item()
    }
    return total_loss, loss_dict

# -------------------------------------------------------------------------
# 5) Dataset: CORAL + BBox
# -------------------------------------------------------------------------
class AuxVisionDataset(Dataset):

    def __init__(self, json_path, mode="train", transform=None, num_regions=11):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.num_regions = num_regions

        self.labels_order = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]
        # Load dictionary
        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "seg_file": datum_dict["seg_file"],
                "label_info": datum_dict["labels"]
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
        seg_file_npy = self.convert_file_path_to_npy(seg_file)
        label_info = data["label_info"]

        # Load 4 modalities
        mod_t1c_file = seg_file_npy.replace("seg", "t1c")
        mod_t1c = np.load(mod_t1c_file)
        mod_t1n_file = seg_file_npy.replace("seg", "t1n")
        mod_t1n = np.load(mod_t1n_file)
        mod_t2f_file = seg_file_npy.replace("seg", "t2f")
        mod_t2f = np.load(mod_t2f_file)
        mod_t2w_file = seg_file_npy.replace("seg", "t2w")
        mod_t2w = np.load(mod_t2w_file)

        if self.transform is not None:
            mod_t1c = self.transform(mod_t1c)
            mod_t1n = self.transform(mod_t1n)
            mod_t2f = self.transform(mod_t2f)
            mod_t2w = self.transform(mod_t2w)

        # Prepare area, extent, solidity, bbox
        # area, extent, solidity => [4], each label an integer in correct range
        area_vals = torch.zeros(4, dtype=torch.long)
        shape_vals = torch.zeros(4, dtype=torch.long)
        satellite_vals = torch.zeros(4, dtype=torch.long)
        region_vals = torch.zeros((4, self.num_regions), dtype=torch.float)

        for i, lbl in enumerate(self.labels_order):
            if lbl in label_info:
                metrics = label_info[lbl]
                area_vals[i] = metrics["area"]
                shape_vals[i] = metrics["shape"]
                satellite_vals[i] = metrics["satellite"]
                region_list = metrics["region"]
                for q in region_list:
                    if q < self.num_regions:
                        region_vals[i,q] = 1.0

        return {
            "t1c": mod_t1c,
            "t1n": mod_t1n,
            "t2f": mod_t2f,
            "t2w": mod_t2w,
            "area_targets": area_vals,         # [4]
            "shape_targets": shape_vals,     # [4]
            "satellite_targets": satellite_vals, # [4]
            "region_targets": region_vals,         # [4,regions]
            "seg_file": seg_file
        }

    def convert_file_path_to_npy(self, image_abs_path):
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"
        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path

# -------------------------------------------------------------------------
# 6) Model with CORAL heads for area/extent/solidity, plus a bounding-box head
# -------------------------------------------------------------------------
class VisionAuxClassifier(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        num_modalities: int = 4,
        area_levels: int = 8,
        shape_levels: int = 7,
        satellite_levels: int = 5,
        num_regions: int = 11,
        use_cls: bool = False,
        projection_strategy: str = "shared",   #  ← NEW
    ):
        super().__init__()
        assert projection_strategy in {"shared", "per_seq", "both"}
        self.strategy = projection_strategy
        self.use_cls  = use_cls
        self.vision_tower = vision_tower

        # ---------------------------------------------
        # Compute feature dimensionality
        # ---------------------------------------------
        if self.use_cls:
            self.cls_dim   = 768 * num_modalities
            self.token_dim = 768 * num_modalities * 2048
        else:
            self.token_dim = 768 * num_modalities * 2048

        # ---------------------------------------------
        # Build heads
        # ---------------------------------------------
        def make_heads(levels, base_dim):
            """
            Helper that returns  ⓐ shared head,  ⓑ ModuleList of per-seq heads
            depending on selected strategy.
            """
            shared = None
            perseq = None

            if self.strategy in {"shared", "both"}:
                # single linear layer that predicts for *all* sequences
                shared = nn.Linear(base_dim, 4 * levels)

            if self.strategy in {"per_seq", "both"}:
                # 4 independent heads – one per sequence
                perseq = nn.ModuleList([
                    nn.Linear(base_dim, levels) for _ in range(4)
                ])

            return shared, perseq

        # area heads (CORAL ⇒ K-1 logits per level)
        self.area_shared,    self.area_perseq    = make_heads(area_levels - 1,
                                                              self.cls_dim if use_cls else self.token_dim)
        # shape heads
        self.shape_shared,   self.shape_perseq   = make_heads(shape_levels,
                                                              self.cls_dim if use_cls else self.token_dim)
        # satellite heads
        self.satellite_shared, self.satellite_perseq = make_heads(satellite_levels,
                                                                  self.cls_dim if use_cls else self.token_dim)
        # region heads
        self.region_shared,  self.region_perseq  = make_heads(num_regions,
                                                              self.token_dim if use_cls else self.token_dim)

        # store constants
        self.area_levels     = area_levels
        self.shape_levels    = shape_levels
        self.satellite_levels = satellite_levels
        self.num_regions     = num_regions

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, mod1, mod2, mod3, mod4):
        B = mod1.size(0)

        # 1) get visual features
        f1, f2, f3, f4 = (self.vision_tower.forward(x) for x in (mod1, mod2, mod3, mod4))

        if self.use_cls:
            # split CLS vs patch tokens
            cls_feats  = torch.cat([f[:, 0]   for f in (f1, f2, f3, f4)], dim=1)  # [B, cls_dim]
            patch_feats = torch.cat([f[:, 1:] for f in (f1, f2, f3, f4)], dim=1)  # [B, cls_dim, 2048]
            patch_feats = patch_feats.view(B, -1)                                 # [B, token_dim]
            area_feats = shape_feats = satellite_feats = cls_feats
            region_feats = patch_feats
        else:
            feats = torch.cat([f1, f2, f3, f4], dim=1).view(B, -1)                # [B, token_dim]
            area_feats = shape_feats = satellite_feats = region_feats = feats

        # 2) helper to compute logits
        def project(shared_layer, perseq_layers, x, levels):
            """
            Returns tensor of shape [B, 4, levels]
            according to chosen projection strategy.
            """
            out = 0
            if shared_layer is not None:     # shared or both
                out += shared_layer(x).view(B, 4, levels)
            if perseq_layers is not None:    # per_seq or both
                per_out = [head(x).view(B, 1, levels) for head in perseq_layers]
                per_out = torch.cat(per_out, dim=1)   # [B,4,levels]
                out = out + per_out if isinstance(out, torch.Tensor) else per_out
            return out

        # 3) compute logits
        area_logits      = project(self.area_shared,      self.area_perseq,
                                   area_feats,      self.area_levels - 1)
        shape_logits     = project(self.shape_shared,     self.shape_perseq,
                                   shape_feats,     self.shape_levels)
        satellite_logits = project(self.satellite_shared, self.satellite_perseq,
                                   satellite_feats, self.satellite_levels)
        region_logits    = project(self.region_shared,    self.region_perseq,
                                   region_feats,    self.num_regions)

        return area_logits, shape_logits, satellite_logits, region_logits


# -------------------------------------------------------------------------
# 7) VisionTrainingArguments (same style as your previous code)
# -------------------------------------------------------------------------
@dataclass
class VisionTrainingArguments:
    model_name_or_path: str = field(
        default="./LaMed/output/LaMed-Phi3-4B-finetune-freeze-viz-0000/hf",
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
    pretrain_vision_model: str = field(default="/local2/amvepa91/M3D/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin",
                                       metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=True, metadata={"help": "Whether to freeze vision tower weights."})
    projection_strategy: str = field(
        default="shared",
        metadata={"help": "Projection-head layout: 'shared' | 'per_seq' | 'both'"}
    )
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    output_dir: str = "./vision_aux_output"
    device: str = "cuda"
    tag: str = ""
    keep_only_region: bool = False
    region_loss: str = "bce"
    use_cls: bool = False


def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_keep_only_region_{args.keep_only_region}_region_loss_{args.region_loss}_use_cls_{args.use_cls}_proj{args.projection_strategy}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(
        log_file=os.path.join(output_dir,
                              f"aux_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_keep_only_region_{args.keep_only_region}_region_loss_{args.region_loss}_use_cls_{args.use_cls}_proj{args.projection_strategy}.log"),
        log_to_console=True
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Example JSON paths for training
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_aux_updated_v2_seed0.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_aux_updated_v2_seed0.json"
    test_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_aux_updated_v2_seed0.json"

    # -----------------------------------------------------------
    # 1) Load the base MLLM with a vision tower
    # -----------------------------------------------------------
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
        num_modalities=4,
        area_levels=8,
        shape_levels=7,
        satellite_levels=5,
        num_regions=11,
        use_cls=args.use_cls,
        projection_strategy=args.projection_strategy,  # ← NEW
    ).to(device)

    # -----------------------------------------------------------
    # 2) Build Datasets / Dataloaders
    # -----------------------------------------------------------
    train_dataset = AuxVisionDataset(train_file, mode="train", num_regions=11)
    val_dataset = AuxVisionDataset(val_file, mode="val", num_regions=11)
    test_dataset = AuxVisionDataset(test_file, mode="test", num_regions=11)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------
    # 3) Optimizer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # -----------------------------------------------------------
    # 4) Training Loop
    # -----------------------------------------------------------
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        area_loss = 0.0
        shape_loss = 0.0
        satellite_loss = 0.0
        region_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            shape_targets = batch["shape_targets"].to(device)     # [B,4]
            satellite_targets = batch["satellite_targets"].to(device) # [B,4]
            region_targets = batch["region_targets"].to(device)         # [B,4,Q]

            optimizer.zero_grad()
            area_logits, shape_logits, satellite_logits, region_logits = model(mod1, mod2, mod3, mod4)

            loss, loss_dict = compute_aux_loss(
                area_logits, shape_logits, satellite_logits, region_logits,
                area_targets, shape_targets, satellite_targets, region_targets,
                K_area=8, K_shape=7, K_satellite=5, keep_only_region=args.keep_only_region,
                region_loss=args.region_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            area_loss += loss_dict["area_loss"]
            shape_loss += loss_dict["shape_loss"]
            satellite_loss += loss_dict["satellite_loss"]
            region_loss += loss_dict["region_loss"]

        avg_train_loss = total_loss / len(train_loader)
        area_loss /= len(train_loader)
        shape_loss /= len(train_loader)
        satellite_loss /= len(train_loader)
        region_loss /= len(train_loader)

        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Area Loss: {area_loss:.4f} - Shape Loss: {shape_loss:.4f} - Satellite Loss: {satellite_loss:.4f} - Region Loss: {region_loss:.4f}")

        # Validation
        val_loss = 0.0
        area_val_loss = 0.0
        shape_val_loss = 0.0
        satellite_val_loss = 0.0
        region_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mod1 = batch["t1c"].to(device)
                mod2 = batch["t1n"].to(device)
                mod3 = batch["t2f"].to(device)
                mod4 = batch["t2w"].to(device)

                area_targets = batch["area_targets"].to(device)
                shape_targets = batch["shape_targets"].to(device)
                satellite_targets = batch["satellite_targets"].to(device)
                region_targets = batch["region_targets"].to(device)

                area_logits, shape_logits, satellite_logits, region_logits = model(mod1, mod2, mod3, mod4)
                loss, loss_dict = compute_aux_loss(
                    area_logits, shape_logits, satellite_logits, region_logits,
                    area_targets, shape_targets, satellite_targets, region_targets,
                    K_area=8, K_shape=7, K_satellite=5, keep_only_region=args.keep_only_region,
                    region_loss=args.region_loss
                )
                area_val_loss += loss_dict["area_loss"]
                shape_val_loss += loss_dict["shape_loss"]
                satellite_val_loss += loss_dict["satellite_loss"]
                region_val_loss += loss_dict["region_loss"]

                val_loss += loss.item()

        area_val_loss /= len(val_loader)
        shape_val_loss /= len(val_loader)
        satellite_val_loss /= len(val_loader)
        region_val_loss /= len(val_loader)
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Area Loss: {area_val_loss:.4f} - Shape Loss: {shape_val_loss:.4f} - Satellite Loss: {satellite_val_loss:.4f} - Region Loss: {region_val_loss:.4f}")

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

    all_area_preds = []
    all_area_tgts  = []
    all_shape_preds = []
    all_shape_tgts  = []
    all_satellite_preds = []
    all_satellite_tgts  = []
    thresh = [-.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1.0, 1.1]
    thresh_iou_list = [dict() for _ in range(4)]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            shape_targets = batch["shape_targets"].to(device)     # [B,4]
            satellite_targets = batch["satellite_targets"].to(device) # [B,4]
            region_targets = batch["region_targets"].to(device)         # [B,4,Q]

            area_logits, shape_logits, satellite_logits, region_logits = model(mod1, mod2, mod3, mod4)

            B = area_logits.size(0)
            area_2d = area_logits.view(B*4, 7)              # => [B*4,9]
            area_pred_1d = coral_predict(area_2d)     # => [B*4]
            area_tgt_1d = area_targets.view(-1)             # => [B*4]
            all_area_preds.append(area_pred_1d.cpu())
            all_area_tgts.append(area_tgt_1d.cpu())

            shape_2d = shape_logits.view(B*4, 7)
            shape_pred_1d = ce_predict(shape_2d)  # => [B*4]
            shape_tgt_1d = shape_targets.view(-1)
            all_shape_preds.append(shape_pred_1d.cpu())
            all_shape_tgts.append(shape_tgt_1d.cpu())

            satellite_2d = satellite_logits.view(B*4, 5)
            satellite_pred_1d = ce_predict(satellite_2d)
            satellite_tgt_1d = satellite_targets.view(-1)
            all_satellite_preds.append(satellite_pred_1d.cpu())
            all_satellite_tgts.append(satellite_tgt_1d.cpu())

            # 4) BBox => IoU
            region_prob = torch.sigmoid(region_logits)  # shape [B,4,Q], in [0..1]
            for thresh_ in thresh:
                region_pred = (region_prob >= thresh_).float()  # hard threshold -> 0/1
                intersection = (region_pred * region_targets).sum(dim=2)  # [B,4]
                union = (region_pred + region_targets - region_pred * region_targets).sum(dim=2)  # [B,4]
                iou = (intersection + 1e-7)/ (union + 1e-7)  # [B,4]
                for label_index in range(4):
                    if thresh_ not in thresh_iou_list[label_index]:
                        thresh_iou_list[label_index][thresh_] = []
                    thresh_iou_list[label_index][thresh_].append(iou.cpu()[:, label_index])

    # stack predictions
    area_preds = torch.cat(all_area_preds).numpy()
    area_tgts  = torch.cat(all_area_tgts).numpy()
    shape_preds = torch.cat(all_shape_preds).numpy()
    shape_tgts  = torch.cat(all_shape_tgts).numpy()
    satellite_preds = torch.cat(all_satellite_preds).numpy()
    satellite_tgts  = torch.cat(all_satellite_tgts).numpy()
    thresh_mean_iou = [dict() for _ in range(4)]
    for label_index in range(4):
        for thresh_ in thresh:
            iou_tensor = torch.cat(thresh_iou_list[label_index][thresh_], dim=0) # shape [N*B]
            thresh_mean_iou[label_index][thresh_] = iou_tensor.mean().item()
    # Simple metrics: Mean Absolute Error for ordinal
    area_mae = mean_absolute_error(area_tgts, area_preds)
    shape_acc = (shape_tgts == shape_preds).mean()
    satellite_acc = (satellite_tgts == satellite_preds).mean()

    logger.info("========== TEST RESULTS ==========")
    logger.info(f"Area MAE:     {area_mae:.4f}")
    logger.info(f"Shape Acc:   {shape_acc:.4f}")
    logger.info(f"Satellite Acc: {satellite_acc:.4f}")
    for label_index in range(4):
        logger.info(f"Label: {label_index}")
        [logger.info(f"\t{thresh_} BBox Mean IoU :{thresh_mean_iou[label_index][thresh_]:.4f}") for thresh_ in thresh]
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
