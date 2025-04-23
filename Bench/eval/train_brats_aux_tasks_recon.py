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

base_path = "/local2/jrgan/NephrologyKG/M3D"
sys.path.append(base_path)
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM


# Precompute a 27x27 dist_matrix
dist_matrix = [[0.0]*27 for _ in range(27)]

coords_3d = []
for idx in range(27):
    r = idx // 9
    c = (idx % 9) // 3
    d = idx % 3
    coords_3d.append((r, c, d))

for i in range(27):
    (r1, c1, d1) = coords_3d[i]
    for j in range(27):
        (r2, c2, d2) = coords_3d[j]
        dist = math.sqrt((r1 - r2)**2 + (c1 - c2)**2 + (d1 - d2)**2)
        dist_matrix[i][j] = dist


def make_soft_label(gt_quadrants, dist_matrix, sigma=1.0):
    label = torch.zeros(27)
    for i in gt_quadrants:
        # If `i` is a PyTorch scalar tensor, do `i_val = i.item()`
        # If `i` is already an int, you can skip this.
        i_val = int(i)  # or i.item() if it's a 1-element tensor

        for j in range(27):
            dist_ij = dist_matrix[i_val][j]
            val = math.exp(- (dist_ij**2) / (2*(sigma**2)))
            label[j] = min(1.0, label[j] + val)
    return label


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


def distance_aware_bce_loss(
    logits,               # shape [B, L, 27]
    gt_quadrants_batch,   # list of length B, each is list of length L, each is list of quadrant indices
    sigma=1.0
):
    """
    Builds a [B, L, 27] soft label for each (b, l) from gt_quadrants_batch
    using RBF adjacency. Then does standard BCEWithLogits with that label.
    Returns a scalar loss.
    """
    B, L, Q = logits.shape
    if Q != 27:
        raise ValueError(f"Expected 27 quadrants, got Q={Q}.")

    # We'll build a tensor of shape [B, L, 27] for soft labels
    soft_labels = torch.zeros(B, L, 27, dtype=torch.float, device=logits.device)

    for b in range(B):
        label_sets_for_b = gt_quadrants_batch[b]  # list of length L
        if len(label_sets_for_b) != L:
            raise ValueError(f"Sample {b} has {len(label_sets_for_b)} label sets, expected {L}.")

        for l in range(L):
            gt_quadrants = label_sets_for_b[l]  # e.g. [0,1,2]
            soft_label_1d = make_soft_label(gt_quadrants, dist_matrix, sigma=sigma)  # shape [27]
            soft_labels[b, l, :] = soft_label_1d

    # Now standard BCEWithLogits
    return F.binary_cross_entropy_with_logits(logits, soft_labels)


def distance_aware_jaccard_loss(
        logits,  # shape [B, L, 27]
        gt_quadrants_batch,  # list of length B, each => list of length L => list of quadrant indices
        dist_matrix,
        sigma=1.0,
        eps=1e-7
):
    """
    1) Builds a [B, L, 27] 'soft label' from RBF adjacency for each sample+label
    2) Uses a 'soft IoU' measure:  1 - intersection/union  => the 'Jaccard loss'

    This typically aligns better with the IoU metric than BCE does.
    """
    B, L, Q = logits.shape
    if Q != 27:
        raise ValueError(f"Expected shape [B,L,27], got Q={Q} instead.")

    # (1) Build the distance-aware labels [B, L, 27]
    soft_labels = torch.zeros(B, L, 27, dtype=torch.float, device=logits.device)

    for b in range(B):
        label_sets_for_b = gt_quadrants_batch[b]  # list of length L
        if len(label_sets_for_b) != L:
            raise ValueError(f"Sample {b} has {len(label_sets_for_b)} label sets, expected {L}.")

        for l in range(L):
            gt_quadrants = label_sets_for_b[l]  # e.g. [0,1,2]
            soft_label_1d = make_soft_label(gt_quadrants, dist_matrix, sigma=sigma)
            soft_labels[b, l, :] = soft_label_1d

    # (2) "Soft IoU" Calculation
    #   - Convert logits -> probabilities via sigmoid
    #   - Sum over the quadrant dimension => shape [B, L]
    probs = torch.sigmoid(logits)  # [B, L, 27]
    intersection = (probs * soft_labels).sum(dim=2)  # [B, L]
    union = (probs + soft_labels - probs * soft_labels).sum(dim=2)  # [B, L]

    # Jaccard = intersection / union, safe with eps
    jaccard = (intersection + eps) / (union + eps)  # [B, L]

    # The "loss" => 1 - average_jaccard
    #   average over all BxL => single scalar
    return 1.0 - jaccard.mean()


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
    """
    Builds a multi-hot label of shape [B, L, 27] with 1's in the
    ground-truth quadrants, then does standard BCEWithLogitsLoss.
    Returns a scalar.
    """
    B, L, Q = logits.shape
    if Q != 27:
        raise ValueError(f"(logits) Expected 27 quadrants, got Q={Q}.")
    B_, L_, Q_ = labels.shape
    if Q_ != 27:
        raise ValueError(f"(labels) Expected 27 quadrants, got Q={Q_}.")

    # Compute BCE with logits
    loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
    return loss_bce


def soft_jaccard_loss(bbox_logits, bbox_targets, eps=1e-7):
    """
    bbox_logits: [B,4,Q]  raw
    bbox_targets: [B,4,Q] 0/1
    Returns a scalar = 1 - mean_jaccard.
    """
    p = torch.sigmoid(bbox_logits)  # => [B,4,Q]
    intersection = (p * bbox_targets).sum(dim=2)  # [B,4]
    union = (p + bbox_targets - p*bbox_targets).sum(dim=2)  # [B,4]
    jaccard = (intersection + eps) / (union + eps)  # [B,4]
    return 1.0 - jaccard.mean()

# -------------------------------------------------------------------------
# 4) Multi-Task Loss: CORAL for area/extent/solidity + Soft Jaccard for bbox
# -------------------------------------------------------------------------
def compute_aux_loss(
    area_logits, extent_logits, solidity_logits, bbox_logits,
    area_targets, extent_targets, solidity_targets, bbox_targets,
    K_area=10, K_extent=6, K_solidity=4, keep_only_bbox=False,
    bbox_loss="jaccard"
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

    if bbox_loss == "dist_jaccard":
        bbox_loss = distance_aware_jaccard_loss(bbox_logits, bbox_targets, dist_matrix)
    elif bbox_loss == "dist_bce":
        bbox_loss = distance_aware_bce_loss(bbox_logits, bbox_targets)
    elif bbox_loss == "bce":
        bbox_loss = bce_loss(bbox_logits, bbox_targets)
    elif bbox_loss == "mse":
        bbox_loss = mse_loss(bbox_logits, bbox_targets)
    elif bbox_loss == "jaccard":
        bbox_loss = soft_jaccard_loss(bbox_logits, bbox_targets)
    else:
        raise ValueError(f"Unknown bbox_loss: {bbox_loss}")

    if keep_only_bbox:
        total_loss = bbox_loss
    else:
        # total_loss = area_loss + extent_loss + solidity_loss + bbox_loss
        total_loss = area_loss + extent_loss + solidity_loss + 0.1 * bbox_loss
    loss_dict = {
        "area_loss": area_loss.item(),
        "extent_loss": extent_loss.item(),
        "solidity_loss": solidity_loss.item(),
        "bbox_loss": bbox_loss.item(),
        
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
        num_modalities=4,
        area_levels=10,
        extent_levels=6,
        solidity_levels=4,
        num_quadrants=27,
        use_cls=False,
        modality_weights=None,
        cross_modality_matrix=None
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.use_cls = use_cls
        self.num_modalities = num_modalities
        
        # Store cross-modality matrix information
        self.cross_modality_matrix = cross_modality_matrix
        if self.cross_modality_matrix is not None:
            # If the vision tower doesn't have cross decoders yet, create them
            if not hasattr(self.vision_tower, 'cross_decoders') or len(self.vision_tower.cross_decoders) == 0:
                self.vision_tower._create_cross_modal_decoders(self.cross_modality_matrix)
        
        if self.use_cls:
            cls_hidden_dim = 768 * num_modalities
            non_cls_hidden_dim = 768 * num_modalities * 2048

            # area => [B,4,(area_levels-1)]
            self.area_head = nn.Linear(cls_hidden_dim, 4 * (area_levels - 1))

            # extent => [B,4,(extent_levels-1)]
            self.extent_head = nn.Linear(cls_hidden_dim, 4 * (extent_levels - 1))

            # solidity => [B,4,(solidity_levels-1)]
            self.solidity_head = nn.Linear(cls_hidden_dim, 4 * (solidity_levels - 1))

            # bbox => [B,4,num_quadrants]
            self.bbox_head = nn.Linear(non_cls_hidden_dim, 4 * num_quadrants)

        else:
            hidden_dim = 768 * num_modalities * 2048

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


    def forward(self, mod1, mod2, mod3, mod4, use_recon=False, mim=True):
        """Unified forward method for both classification and reconstruction tasks
        
        Args:
            mod1, mod2, mod3, mod4: Input images for different modalities [B, C, D, H, W]
            use_recon: Whether to use reconstruction (masking)
            
        Returns:
            If use_recon=False:
                area_logits, extent_logits, solidity_logits, bbox_logits
            If use_recon=True:
                Dictionary with classification and reconstruction outputs
        """
        B = mod1.size(0)
        
        if use_recon:
            if mim:
                # Process all modalities with MIM
                multimodal_outputs = self.vision_tower.forward_mim_multimodal(mod1, mod2, mod3, mod4, apply_mask=True)
                
            else:
                # Get multimodal features with masking for reconstruction
                multimodal_outputs = self.vision_tower(mod1, apply_mask=True, multimodal=True, 
                                                     mod2=mod2, mod3=mod3, mod4=mod4)
                
            # Get features for classification task
            feats1 = multimodal_outputs['modality_results'][0]['features']
            feats2 = multimodal_outputs['modality_results'][1]['features']
            feats3 = multimodal_outputs['modality_results'][2]['features']
            feats4 = multimodal_outputs['modality_results'][3]['features']
        else:
            # Standard forward pass without masking
            feats1 = self.vision_tower(mod1)
            feats2 = self.vision_tower(mod2)
            feats3 = self.vision_tower(mod3)
            feats4 = self.vision_tower(mod4)
        
        # Process features for classification tasks (same for both modes)
        if self.use_cls:
            # Concatenate cls features and then non-cls features
            cls_feats = torch.cat([feats1[:, 0], feats2[:, 0], feats3[:, 0], feats4[:, 0]], dim=1)
            cls_feats = cls_feats.view(B, -1)  # [B, 768*4]
            non_cls_feats = torch.cat([feats1[:, 1:], feats2[:, 1:], feats3[:, 1:], feats4[:, 1:]], dim=1)
            non_cls_feats = non_cls_feats.view(B, -1)  # [B, 768*4*2048]
            area_feats = cls_feats
            extent_feats = cls_feats
            solidity_feats = cls_feats
            bbox_feats = non_cls_feats
        else:
            # Concatenate features from all modalities
            feats = torch.cat([feats1, feats2, feats3, feats4], dim=1)
            feats = feats.view(B, -1)
            area_feats = feats
            extent_feats = feats
            solidity_feats = feats
            bbox_feats = feats
            
        # Compute classification logits
        area_raw = self.area_head(area_feats)
        area_logits = area_raw.view(B, 4, (self.area_levels - 1))
        
        extent_raw = self.extent_head(extent_feats)
        extent_logits = extent_raw.view(B, 4, (self.extent_levels - 1))
        
        solidity_raw = self.solidity_head(solidity_feats)
        solidity_logits = solidity_raw.view(B, 4, (self.solidity_levels - 1))
        
        bbox_raw = self.bbox_head(bbox_feats)
        bbox_logits = bbox_raw.view(B, 4, self.num_quadrants)
        
        if use_recon:
            # Return both classification and reconstruction outputs
            results = {
                'classification': {
                    'area_logits': area_logits,
                    'extent_logits': extent_logits,
                    'solidity_logits': solidity_logits,
                    'bbox_logits': bbox_logits
                },
                'reconstruction': {
                    'original_patches_list': [x['original_patches'] for x in multimodal_outputs['modality_results']],
                    'reconstructed_patches_list': [x['reconstructed_patches'] for x in multimodal_outputs['modality_results']],
                    'mask_indices_list': [x['mask_indices'] for x in multimodal_outputs['modality_results']],
                    # 'masks': multimodal_outputs['masks']
                }
            }
            # Include cross-modal reconstruction if available
            if 'cross_reconstructed' in multimodal_outputs:
                results['reconstruction']['cross_reconstructed'] = multimodal_outputs['cross_reconstructed']
                
            return results
        else:
            # Return only classification outputs
            return area_logits, extent_logits, solidity_logits, bbox_logits
            
# -------------------------------------------------------------------------
# 7) VisionTrainingArguments (same style as your previous code)
# -------------------------------------------------------------------------
@dataclass
class VisionTrainingArguments:
    model_name_or_path: str = field(
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        # default="./LaMed/output/LaMed-Phi3-4B-finetune-freeze-viz-0000/hf",
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

    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    output_dir: str = "./vision_aux_output_recon"
    device: str = "cuda"
    tag: str = ""
    keep_only_bbox: bool = False
    bbox_loss: str = "bce"
    use_cls: bool = True
    # arguments for reconstruction
    enable_reconstruction: bool = field(default=True, metadata={"help": "Whether to enable reconstruction task"})
    reconstruction_weight: float = field(default=0.5, metadata={"help": "Weight for reconstruction loss"})
    mask_ratio: float = field(default=0.3, metadata={"help": "Ratio of patches to mask for reconstruction"})
    masked_image_modeling: bool = field(default=False, metadata={"help": "Masked Image Modeling or Masked Feature Learning"})

    

def compute_reconstruction_metrics(original_patches_list, reconstructed_patches_list, mask_indices_list):
    """
    Compute PSNR for reconstructed patches across modalities
    
    Args:
        original_patches_list: List of original patches for each modality
        reconstructed_patches_list: List of reconstructed patches for each modality
        mask_indices_list: List of mask indices for each batch and modality
    
    Returns:
        metrics: Dictionary with PSNR metrics
    """
    metrics = {}
    all_psnr_values = []
    
    # Process each modality
    for mod_idx, (orig_patches, recon_patches, mask_indices) in enumerate(
            zip(original_patches_list, reconstructed_patches_list, mask_indices_list)):
        psnr_values = []
        # Process each batch
        B = orig_patches.shape[0]  # Should be consistent across all modalities
        
        for b in range(B):
            if b < len(mask_indices):  # Check if this batch has mask indices
                masked_idx = mask_indices[b]
                if len(masked_idx) > 0:
                    # Get original and reconstructed patches for this batch
                    orig = orig_patches[b, masked_idx]
                    recon = recon_patches[b, masked_idx]
                    
                    # Compute MSE
                    mse = F.mse_loss(recon, orig)
                    mse = torch.clamp(mse, min=1e-8)  # Avoid division by zero
                    
                    # Compute PSNR
                    max_signal = torch.max(orig).clamp(min=1e-6)
                    psnr = 20 * torch.log10(max_signal / torch.sqrt(mse))
                    psnr_values.append(psnr.item())
        
        # Average PSNR for this modality
        if psnr_values:
            metrics[f'psnr_mod{mod_idx+1}'] = sum(psnr_values) / len(psnr_values)
            all_psnr_values.extend(psnr_values)
        else:
            metrics[f'psnr_mod{mod_idx+1}'] = 0.0
            
    # Average PSNR across all modalities
    metrics['psnr_avg'] = sum(all_psnr_values) / len(all_psnr_values) if all_psnr_values else 0.0
    
    return metrics
    

def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_keep_only_bbox_{args.keep_only_bbox}_bbox_loss_{args.bbox_loss}_use_cls_{args.use_cls}_mask{args.mask_ratio}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(
        log_file=os.path.join(output_dir,
                              f"aux_model_name_{os.path.basename(args.pretrain_vision_model)}_freeze_vision_{args.freeze_vision_tower}_epochs_{args.num_epochs}_keep_only_bbox_{args.keep_only_bbox}_bbox_loss_{args.bbox_loss}_use_cls_{args.use_cls}.log"),
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
    
    # Add cross_modality_matrix configuration to vision tower
    cross_modality_matrix = torch.zeros((4, 4))
    for i in range(4):
        cross_modality_matrix[i][i] = 0.0  # No self-reconstruction in cross-modal (handled separately)
    # Focus on T2F to other modalities reconstruction  
    cross_modality_matrix[2, 0] = 1.5  # T2f -> T1c
    cross_modality_matrix[2, 1] = 1.5  # T2f -> T1n 
    cross_modality_matrix[2, 3] = 1.5  # T2f -> T2w
    cross_modality_matrix[0, 1] = 1.0  # T2f -> T1c
    cross_modality_matrix[0, 2] = 1.0  # T2f -> T1n 
    cross_modality_matrix[0, 3] = 1.0  # T2f -> T2w
    
    # Define modality weights for self-reconstruction (give T2F higher weight)
    modality_weights = [1.0, 1.0, 2.0, 1.0]
    vision_tower.modality_weights = modality_weights
    
    # Add these parameters to vision_tower config
    if not hasattr(vision_tower, 'config'):
        class TempConfig:
            pass
        vision_tower.config = TempConfig()
    
    vision_tower.num_modalities = 4
    vision_tower.cross_modality_matrix = cross_modality_matrix
    
    # Update vision tower to create cross-modal decoders
    if hasattr(vision_tower, '_create_cross_modal_decoders'):
        vision_tower._create_cross_modal_decoders(cross_modality_matrix)


    if args.use_cls:
        vision_tower.select_feature = 'cls_patch'
        logger.info("Using [CLS] token during training.")

    # Set mask ratio for reconstruction if enabled
    if args.enable_reconstruction:
        vision_tower.mask_ratio = args.mask_ratio
        logger.info(f"Setting mask ratio to {args.mask_ratio} for reconstruction task")
    
    # Freeze if requested
    if args.freeze_vision_tower:
        for param in vision_tower.parameters():
            param.requires_grad = False
        logger.info("Vision tower is frozen.")
    else:
        unfrozen_params = 0
        total_params = 0
        for name, param in vision_tower.named_parameters():
            total_params += param.numel()
            # Check if this parameter belongs to one of the blocks to unfreeze
            should_unfreeze = False
            for block_idx in [10,11]:
                if f"blocks.{block_idx}" in name:
                    should_unfreeze = True
                    break
            # Also unfreeze the final normalization layer
            if "norm" in name and "blocks" not in name:  # Only the final norm, not block norms
                should_unfreeze = True
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_params += param.numel()
        print(f"Unfrozen {unfrozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Percentage unfrozen: {unfrozen_params / total_params * 100:.2f}%")
    
    # Build the multi-task model
    model = VisionAuxClassifier(
        vision_tower=vision_tower,
        num_modalities=4,
        area_levels=10,      # 10 ordinal categories
        extent_levels=6,     # 6 ordinal categories
        solidity_levels=4,   # 4 ordinal categories
        num_quadrants=27,    # 3x3x3 bounding box
        use_cls=args.use_cls,
        modality_weights=modality_weights,
        cross_modality_matrix=cross_modality_matrix
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
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # -----------------------------------------------------------
    # 4) Training Loop
    # -----------------------------------------------------------
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        area_loss = 0.0
        extent_loss = 0.0
        solidity_loss = 0.0
        bbox_loss = 0.0
        recon_loss = 0.0
        self_recon_loss = 0.0
        cross_recon_loss = 0.0
        task_loss = 0.0
        psnr_avg = 0.0 
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mod1 = batch["t1c"].to(device)  # T1-contrast
            mod2 = batch["t1n"].to(device)  # T1-native
            mod3 = batch["t2f"].to(device)  # T2-FLAIR
            mod4 = batch["t2w"].to(device)  # T2-weighted
        
            area_targets = batch["area_targets"].to(device)         # [B,4]
            extent_targets = batch["extent_targets"].to(device)     # [B,4]
            solidity_targets = batch["solidity_targets"].to(device) # [B,4]
            bbox_targets = batch["bbox_targets"].to(device)         # [B,4,Q]
        
            optimizer.zero_grad()
            
            if args.enable_reconstruction:
                # Forward pass with reconstruction
                results = model(mod1, mod2, mod3, mod4, use_recon=True, mim=args.masked_image_modeling)
                
                # Get classification outputs
                area_logits = results['classification']['area_logits']
                extent_logits = results['classification']['extent_logits']
                solidity_logits = results['classification']['solidity_logits']
                bbox_logits = results['classification']['bbox_logits']
                
                # Get reconstruction outputs
                original_patches_list = results['reconstruction']['original_patches_list']
                reconstructed_patches_list = results['reconstruction']['reconstructed_patches_list']
                mask_indices_list = results['reconstruction']['mask_indices_list']
                
                # Cross-modal reconstruction outputs - may be missing if mim=False
                cross_reconstructed_patches_dict = {}
                if 'cross_reconstructed' in results['reconstruction']:
                    cross_reconstructed_patches_dict = results['reconstruction']['cross_reconstructed']
                
                # Compute reconstruction loss depending on mode
                if args.masked_image_modeling:
                    # Compute reconstruction loss using vision tower's method
                    rec_loss, rec_loss_dict = model.vision_tower.compute_multimodal_mim_loss(
                        original_patches_list,
                        reconstructed_patches_list,
                        cross_reconstructed_patches_dict,
                        mask_indices_list
                    )
                else:
                    # Compute reconstruction loss using compute_cross_modality_reconstruction_loss
                    rec_loss, rec_loss_dict = model.vision_tower.compute_cross_modality_reconstruction_loss(
                        original_patches_list,
                        reconstructed_patches_list,
                        cross_reconstructed_patches_dict,
                        mask_indices_list
                    )
                
                # Main task loss (classification)
                main_loss, loss_dict = compute_aux_loss(
                    area_logits, extent_logits, solidity_logits, bbox_logits,
                    area_targets, extent_targets, solidity_targets, bbox_targets,
                    K_area=10, K_extent=6, K_solidity=4, keep_only_bbox=args.keep_only_bbox,
                    bbox_loss=args.bbox_loss
                )
                
                # Combined loss
                loss = main_loss + args.reconstruction_weight * rec_loss
                
                # Update metrics
                recon_loss += rec_loss.item()
                self_recon_loss += rec_loss_dict.get('self_reconstruction', 0.0)
                cross_recon_loss += rec_loss_dict.get('cross_reconstruction', 0.0)
                task_loss += main_loss.item()
            else:
                # Standard forward pass without reconstruction
                area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)
                
                loss, loss_dict = compute_aux_loss(
                    area_logits, extent_logits, solidity_logits, bbox_logits,
                    area_targets, extent_targets, solidity_targets, bbox_targets,
                    K_area=10, K_extent=6, K_solidity=4, keep_only_bbox=args.keep_only_bbox,
                    bbox_loss=args.bbox_loss
                )
                task_loss += loss.item()
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            area_loss += loss_dict["area_loss"]
            extent_loss += loss_dict["extent_loss"]
            solidity_loss += loss_dict["solidity_loss"]
            bbox_loss += loss_dict["bbox_loss"]

        # Calculate averages over the entire epoch
        avg_train_loss = total_loss / len(train_loader)
        avg_task_loss = task_loss / len(train_loader)
        avg_area_loss = area_loss / len(train_loader)
        avg_extent_loss = extent_loss / len(train_loader)
        avg_solidity_loss = solidity_loss / len(train_loader)
        avg_bbox_loss = bbox_loss / len(train_loader)

        log_message = f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Area: {avg_area_loss:.4f} - Extent: {avg_extent_loss:.4f} - Solidity: {avg_solidity_loss:.4f} - BBox: {avg_bbox_loss:.4f}"
        
        if args.enable_reconstruction:
            avg_recon_loss = recon_loss / len(train_loader)
            avg_self_recon = self_recon_loss / len(train_loader)
            avg_cross_recon = cross_recon_loss / len(train_loader)
            log_message += f" - Recon Loss: {avg_recon_loss:.4f} (Self: {avg_self_recon:.4f}, Cross: {avg_cross_recon:.4f})"
        
        logger.info(log_message)
        
        # Validation
        model.eval()
        val_loss = 0.0
        area_val_loss = 0.0
        extent_val_loss = 0.0
        solidity_val_loss = 0.0
        bbox_val_loss = 0.0
        val_recon_loss = 0.0
        val_self_recon = 0.0
        val_cross_recon = 0.0
        val_task_loss = 0.0
        val_psnr_avg = 0.0
        
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
                
                if args.enable_reconstruction:
                    # Forward pass with reconstruction
                    results = model(mod1, mod2, mod3, mod4, use_recon=True, mim=args.masked_image_modeling)
                
                    # Get classification outputs
                    area_logits = results['classification']['area_logits']
                    extent_logits = results['classification']['extent_logits']
                    solidity_logits = results['classification']['solidity_logits']
                    bbox_logits = results['classification']['bbox_logits']
                    
                    # Get reconstruction outputs
                    original_patches_list = results['reconstruction']['original_patches_list']
                    reconstructed_patches_list = results['reconstruction']['reconstructed_patches_list']
                    mask_indices_list = results['reconstruction']['mask_indices_list']
                    
                    # Cross-modal reconstruction outputs - may be missing if mim=False
                    cross_reconstructed_patches_dict = {}
                    if 'cross_reconstructed' in results['reconstruction']:
                        cross_reconstructed_patches_dict = results['reconstruction']['cross_reconstructed']
                    
                    # Calculate reconstruction metrics
                    recon_metrics = compute_reconstruction_metrics(
                        original_patches_list, 
                        reconstructed_patches_list, 
                        mask_indices_list
                    )
                    val_psnr_avg += recon_metrics['psnr_avg']
                    
                    # Calculate reconstruction loss
                    if args.masked_image_modeling:
                        rec_loss, rec_loss_dict = model.vision_tower.compute_multimodal_mim_loss(
                            original_patches_list,
                            reconstructed_patches_list,
                            cross_reconstructed_patches_dict,
                            mask_indices_list
                        )
                    else:
                        rec_loss, rec_loss_dict = model.vision_tower.compute_cross_modality_reconstruction_loss(
                            original_patches_list,
                            reconstructed_patches_list,
                            cross_reconstructed_patches_dict,
                            mask_indices_list
                        )
                    
                    val_recon_loss += rec_loss.item()
                    val_self_recon += rec_loss_dict.get('self_reconstruction', 0.0)
                    val_cross_recon += rec_loss_dict.get('cross_reconstruction', 0.0)
                else:
                    # Standard forward pass without reconstruction
                    area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)
                
                # Calculate classification loss
                main_loss, loss_dict = compute_aux_loss(
                    area_logits, extent_logits, solidity_logits, bbox_logits,
                    area_targets, extent_targets, solidity_targets, bbox_targets,
                    K_area=10, K_extent=6, K_solidity=4, keep_only_bbox=args.keep_only_bbox,
                    bbox_loss=args.bbox_loss
                )
                
                # Combined loss (if reconstruction enabled)
                if args.enable_reconstruction:
                    loss = main_loss + args.reconstruction_weight * rec_loss
                else:
                    loss = main_loss
                
                area_val_loss += loss_dict["area_loss"]
                extent_val_loss += loss_dict["extent_loss"]
                solidity_val_loss += loss_dict["solidity_loss"]
                bbox_val_loss += loss_dict["bbox_loss"]
                val_task_loss += main_loss.item()
                val_loss += loss.item()

        # Calculate validation averages
        area_val_loss /= len(val_loader)
        extent_val_loss /= len(val_loader)
        solidity_val_loss /= len(val_loader)
        bbox_val_loss /= len(val_loader)
        val_loss /= len(val_loader)
        val_task_loss /= len(val_loader)
        
        val_log_message = f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Task Loss: {val_task_loss:.4f} - Area: {area_val_loss:.4f} - Extent: {extent_val_loss:.4f} - Solidity: {solidity_val_loss:.4f} - BBox: {bbox_val_loss:.4f}"
        
        if args.enable_reconstruction:
            val_recon_loss /= len(val_loader)
            val_self_recon /= len(val_loader)
            val_cross_recon /= len(val_loader)
            val_psnr_avg /= len(val_loader)
            val_log_message += f" - Recon Loss: {val_recon_loss:.4f} (Self: {val_self_recon:.4f}, Cross: {val_cross_recon:.4f}) - PSNR: {val_psnr_avg:.2f} dB"
            
        logger.info(val_log_message)
        
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
    thresh = [-.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1.0, 1.1]
    thresh_iou_list = [dict() for _ in range(4)]
    
    test_recon_psnr = 0.0
    test_recon_loss = 0.0
    test_samples = 0

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

            # For reconstruction evaluation (if enabled)
            if args.enable_reconstruction:
                results = model(mod1, mod2, mod3, mod4, use_recon=True, mim=args.masked_image_modeling)
                area_logits = results['classification']['area_logits']
                extent_logits = results['classification']['extent_logits']
                solidity_logits = results['classification']['solidity_logits']
                bbox_logits = results['classification']['bbox_logits']
                
                # Get reconstruction outputs for metrics
                original_patches_list = results['reconstruction']['original_patches_list']
                reconstructed_patches_list = results['reconstruction']['reconstructed_patches_list']
                mask_indices_list = results['reconstruction']['mask_indices_list']
                
                # Calculate reconstruction metrics
                recon_metrics = compute_reconstruction_metrics(
                    original_patches_list, 
                    reconstructed_patches_list, 
                    mask_indices_list
                )
                test_recon_psnr += recon_metrics['psnr_avg']
                
                # Calculate reconstruction loss
                cross_reconstructed_patches_dict = {}
                if 'cross_reconstructed' in results['reconstruction']:
                    cross_reconstructed_patches_dict = results['reconstruction']['cross_reconstructed']
                    
                rec_loss, _ = model.vision_tower.compute_cross_modality_reconstruction_loss(
                    original_patches_list,
                    reconstructed_patches_list,
                    cross_reconstructed_patches_dict,
                    mask_indices_list
                )
                test_recon_loss += rec_loss.item()
            else:
                # Standard forward pass without reconstruction
                area_logits, extent_logits, solidity_logits, bbox_logits = model(mod1, mod2, mod3, mod4)
                    
            # 1) Area CORAL => [B,4,9]
            B = area_logits.size(0)
            test_samples += B
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
            bbox_prob = torch.sigmoid(bbox_logits)  # shape [B,4,Q], in [0..1]
            for thresh_ in thresh:
                bbox_pred = (bbox_prob >= thresh_).float()  # hard threshold -> 0/1
                intersection = (bbox_pred * bbox_targets).sum(dim=2)  # [B,4]
                union = (bbox_pred + bbox_targets - bbox_pred * bbox_targets).sum(dim=2)  # [B,4]
                iou = (intersection + 1e-7)/ (union + 1e-7)  # [B,4]
                for label_index in range(4):
                    if thresh_ not in thresh_iou_list[label_index]:
                        thresh_iou_list[label_index][thresh_] = []
                    thresh_iou_list[label_index][thresh_].append(iou.cpu()[:, label_index])

    # stack predictions
    area_preds = torch.cat(all_area_preds).numpy()
    area_tgts  = torch.cat(all_area_tgts).numpy()
    extent_preds = torch.cat(all_extent_preds).numpy()
    extent_tgts  = torch.cat(all_extent_tgts).numpy()
    solidity_preds = torch.cat(all_solidity_preds).numpy()
    solidity_tgts  = torch.cat(all_solidity_tgts).numpy()
    
    # Compute IoU metrics
    thresh_mean_iou = [dict() for _ in range(4)]
    for label_index in range(4):
        for thresh_ in thresh:
            iou_tensor = torch.cat(thresh_iou_list[label_index][thresh_], dim=0) # shape [N*B]
            thresh_mean_iou[label_index][thresh_] = iou_tensor.mean().item()
            
    # Simple metrics: Mean Absolute Error for ordinal
    area_mae = mean_absolute_error(area_tgts, area_preds)
    extent_mae = mean_absolute_error(extent_tgts, extent_preds)
    solidity_mae = mean_absolute_error(solidity_tgts, solidity_preds)

    logger.info("========== TEST RESULTS ==========")
    logger.info(f"Area MAE:     {area_mae:.4f}")
    logger.info(f"Extent MAE:   {extent_mae:.4f}")
    logger.info(f"Solidity MAE: {solidity_mae:.4f}")
    
    for label_index in range(4):
        logger.info(f"Label: {label_index}")
        [logger.info(f"\t{thresh_} BBox Mean IoU :{thresh_mean_iou[label_index][thresh_]:.4f}") for thresh_ in thresh]
    
    # Log reconstruction metrics if enabled
    if args.enable_reconstruction:
        avg_test_recon_loss = test_recon_loss / len(test_loader)
        avg_test_recon_psnr = test_recon_psnr / len(test_loader)
        logger.info(f"Reconstruction Loss: {avg_test_recon_loss:.4f}")
        logger.info(f"Reconstruction PSNR: {avg_test_recon_psnr:.2f} dB")
        
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
