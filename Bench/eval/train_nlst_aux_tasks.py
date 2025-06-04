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
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM


labels_order = [
            "NA",
            "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
            "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
            "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
            "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
            "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
            "Non-calcified micronodule(s) (opacity < 4 mm diameter)",
            "Benign lung nodule(s) (benign calcification)",
            "Atelectasis, segmental or greater",
            "Pleural thickening or effusion",
            "Non-calcified hilar/mediastinal adenopathy or mass (>= 10 mm on short axis)",
            "Chest wall abnormality",
            "Consolidation",
            "Emphysema",
            "Significant cardiovascular abnormality",
            "Significant cardiovascular abnormality",
            "Reticular/reticulonodular opacities",
            "6 or more nodules, not suspicious for cancer (opacity >= 4 mm)",
            "Other potentially significant abnormality above the diaphragm",
            "Other potentially significant abnormality above the diaphragm",
            "Other potentially significant abnormality above the diaphragm",
            "Other potentially significant abnormality above the diaphragm",
            "Other potentially significant abnormality below the diaphragm",
            "Other potentially significant abnormality below the diaphragm",
            "Other potentially significant abnormality below the diaphragm",
            "Other minor abnormality noted",
            "Other minor abnormality noted",
            "Other minor abnormality noted"
        ]


sct_ab_code_dict = {
    51: "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
    52: "Non-calcified micronodule(s) (opacity < 4 mm diameter)",
    53: "Benign lung nodule(s) (benign calcification)",
    54: "Atelectasis, segmental or greater",
    55: "Pleural thickening or effusion",
    56: "Non-calcified hilar/mediastinal adenopathy or mass (>= 10 mm on short axis)",
    57: "Chest wall abnormality",
    58: "Consolidation",
    59: "Emphysema",
    60: "Significant cardiovascular abnormality",
    61: "Reticular/reticulonodular opacities",
    62: "6 or more nodules, not suspicious for cancer (opacity >= 4 mm)",
    63: "Other potentially significant abnormality above the diaphragm",
    64: "Other potentially significant abnormality below the diaphragm",
    65: "Other minor abnormality noted",
    # .M, .N, etc. can be mapped as needed. If numeric codes are stored as strings, adjust keys accordingly
}

sct_epi_loc_dict = {
    1: "Right Upper Lobe",
    2: "Right Middle Lobe",
    3: "Right Lower Lobe",
    4: "Left Upper Lobe",
    5: "Lingula",
    6: "Left Lower Lobe",
    8: "Other (see comments)",
    # .N => "Not Applicable", etc.
}

sct_margins_dict = {
    1: "Spiculated (Stellate)",
    2: "Smooth",
    3: "Poorly defined",
    9: "Unable to determine",
    # .N => "Not applicable", etc.
}

sct_pre_att_dict = {
    1: "Soft Tissue",
    2: "Ground glass",
    3: "Mixed",
    4: "Fluid/water",
    6: "Fat",
    7: "Other",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_attn_dict = {
    1: "No interval change in attenuation",
    2: "Yes, suspicious change in attenuation",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_gwth_dict = {
    1: "No interval growth",
    2: "Yes, interval growth",
    9: "Unable to determine"
    # .N => "Not applicable"
}

sct_ab_invg_dict = {
    1: "No further investigation needed",
    2: "Yes, warrants further investigation",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable"
}

sct_ab_preexist_dict = {
    1: "No",
    2: "Yes",
    9: "Unable to determine"
    # .M => "Missing"
}

abnormality_type_map = {sct_ab_code_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_ab_code_dict.keys()))}
location_map = {sct_epi_loc_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_epi_loc_dict.keys()))}
margins_map = {sct_margins_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_margins_dict.keys()))}
pre_att_map = {sct_pre_att_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_pre_att_dict.keys()))}
interval_change_map = {sct_ab_attn_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_ab_attn_dict.keys()))}
interval_growth_map = {sct_ab_gwth_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_ab_gwth_dict.keys()))}
further_investigation_map = {sct_ab_invg_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_ab_invg_dict.keys()))}
ab_preexist_map = {sct_ab_preexist_dict.get(key, "NA"): index for index, key in enumerate(["NA"] + sorted(sct_ab_preexist_dict.keys()))}


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


def get_npy_path(volume_path, img_root="/local/amvepa91/nlst_npy"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels.long())

def argmax_ignore_nan(logits: torch.Tensor):
    """[B,L,C] -> [B,L] int64; logits can be float32/16."""
    return torch.argmax(logits, dim=-1)

@torch.no_grad()
def accuracy(pred: torch.Tensor, tgt: torch.Tensor):
    """Both [N] int64; returns float in [0,1]."""
    valid = torch.isfinite(tgt)
    if valid.sum() == 0:     # no valid labels
        return 0.0
    return (pred[valid] == tgt[valid]).float().mean().item()

# ----------------------------------------------------------------------
#  gather labels over a dataset to build majority / mean baselines
# ----------------------------------------------------------------------
def collect_labels(loader, device):
    """
    Returns dict task -> list[torch.Tensor]  (concatenated over dataset)
    Each list entry is 1-D.
    """
    store = collections.defaultdict(list)
    for batch in tqdm(loader, desc="[collect]", leave=False):
        for k, v in batch["label_dict"].items():
            store[k].append(v.flatten().to(device))
    return {k: torch.cat(v) for k, v in store.items()}

def build_baseline(train_labels):
    """
    From the collected train labels compute
      * majority class for categorical tasks
      * mean   value  for regression tasks
    Returns two dicts.
    """
    majority = {}
    mean_val = {}
    for k, v in train_labels.items():
        if "diameter" in k:                  # regression task
            val = v[torch.isfinite(v)].float()
            mean_val[k] = val.mean().item() if val.numel() else 0.0
        else:                                # categorical
            vals = v[torch.isfinite(v)].long()
            mode = torch.mode(vals, keepdim=False).values.item() if vals.numel() else 0
            majority[k] = mode
    return majority, mean_val

@torch.no_grad()
def eval_baseline(loader, majority, mean_val, device):
    """
    Evaluate the constant predictors on a dataloader; returns dict of metrics.
    """
    tot_correct = collections.defaultdict(int)
    tot_counts  = collections.defaultdict(int)
    mse_sum     = collections.defaultdict(float)

    for batch in loader:
        for k, tgt in batch["label_dict"].items():
            tgt = tgt.to(device).flatten()
            if "diameter" in k:
                pred = torch.full_like(tgt, mean_val[k], dtype=torch.float)
                mse_sum[k] += ((pred - tgt) ** 2)[torch.isfinite(tgt)].sum().item()
                tot_counts[k] += torch.isfinite(tgt).sum().item()
            else:
                pred = torch.full_like(tgt, majority[k], dtype=torch.long)
                mask = torch.isfinite(tgt)
                tot_correct[k] += (pred[mask] == tgt[mask]).sum().item()
                tot_counts[k]  += mask.sum().item()

    out = {}
    for k in tot_counts:
        if "diameter" in k:
            out[k + "_mse"] = mse_sum[k] / max(1, tot_counts[k])
        else:
            out[k + "_acc"] = tot_correct[k] / max(1, tot_counts[k])
    return out


def mse_ignore_nan(pred: torch.Tensor,
                   target: torch.Tensor,
                   reduction: str = "mean") -> torch.Tensor:
    """
    pred   : arbitrary shape, float
    target : same shape as pred, may contain NaNs
    reduction : "mean" | "sum" | "none"
    Returns 0 (if every element is NaN) so the rest of the loss can still back-prop.
    """
    mask = torch.isfinite(target)          # False where NaN or ±Inf
    if reduction == "none":
        out = (pred - target).pow(2)
        out[~mask] = 0.0                   # keep shape, zero-out ignored slots
        return out

    if mask.sum() == 0:                    # all targets are invalid
        return torch.tensor(0.0,
                            device=pred.device,
                            dtype=pred.dtype,
                            requires_grad=pred.requires_grad)

    diff2 = (pred[mask] - target[mask]).pow(2)
    return diff2.mean() if reduction == "mean" else diff2.sum()


def compute_aux_loss(abnormality_type_logits, preexist_logits, interval_change_logits, interval_growth_logits,
                     location_logits, further_investigation_logits, margins_logits, pre_att_logits,
                     longest_diameter_reg_logits, longest_perp_diameter_reg_logits, abnormality_type_labels,
                     preexist_labels, location_labels, interval_change_labels, interval_growth_labels,
                     further_investigation_labels, margins_labels, pre_att_labels,
                     longest_diameter_labels, longest_perp_diameter_labels,
                     num_labels=28):
    B = abnormality_type_logits.size(0)

    abnormality_type_2d = abnormality_type_logits.view(B * num_labels, -1)
    abnormality_type_tgt_1d = abnormality_type_labels.view(B * num_labels)
    abnormality_type_loss = ce_loss(abnormality_type_2d, abnormality_type_tgt_1d)

    preexist_2d = preexist_logits.view(B * num_labels, -1)
    preexist_tgt_1d = preexist_labels.view(B * num_labels)
    preexist_loss = ce_loss(preexist_2d, preexist_tgt_1d)

    location_2d = location_logits.view(B * num_labels, -1)
    location_tgt_1d = location_labels.view(B * num_labels)
    location_loss = ce_loss(location_2d, location_tgt_1d)

    interval_change_2d = interval_change_logits.view(B * num_labels, -1)
    interval_change_tgt_1d = interval_change_labels.view(B * num_labels)
    interval_change_loss = ce_loss(interval_change_2d, interval_change_tgt_1d)

    interval_growth_2d = interval_growth_logits.view(B * num_labels, -1)
    interval_growth_tgt_1d = interval_growth_labels.view(B * num_labels)
    interval_growth_loss = ce_loss(interval_growth_2d, interval_growth_tgt_1d)

    further_investigation_2d = further_investigation_logits.view(B * num_labels, -1)
    further_investigation_tgt_1d = further_investigation_labels.view(B * num_labels)
    further_investigation_loss = ce_loss(further_investigation_2d, further_investigation_tgt_1d)

    margins_2d = margins_logits.view(B * num_labels, -1)
    margins_tgt_1d = margins_labels.view(B * num_labels)
    margins_loss = ce_loss(margins_2d, margins_tgt_1d)

    pre_att_2d = pre_att_logits.view(B * num_labels, -1)
    pre_att_tgt_1d = pre_att_labels.view(B * num_labels)
    pre_att_loss = ce_loss(pre_att_2d, pre_att_tgt_1d)

    longest_diameter_reg_2d = longest_diameter_reg_logits.view(B * num_labels)
    longest_diameter_tgt_1d = longest_diameter_labels.view(B * num_labels)
    longest_diameter_loss = mse_ignore_nan(longest_diameter_reg_2d, longest_diameter_tgt_1d)

    longest_perp_diameter_reg_2d = longest_perp_diameter_reg_logits.view(B * num_labels)
    longest_perp_diameter_tgt_1d = longest_perp_diameter_labels.view(B * num_labels)
    longest_perp_diameter_loss = mse_ignore_nan(longest_perp_diameter_reg_2d, longest_perp_diameter_tgt_1d)

    total_loss = abnormality_type_loss + preexist_loss + location_loss + interval_change_loss + interval_growth_loss + \
        further_investigation_loss + margins_loss + pre_att_loss
    loss_dict = {
        "abnormality_type_loss": abnormality_type_loss.item(),
        "preexist_loss": preexist_loss.item(),
        "location_loss": location_loss.item(),
        "interval_change_loss": interval_change_loss.item(),
        "interval_growth_loss": interval_growth_loss.item(),
        "further_investigation_loss": further_investigation_loss.item(),
        "margins_loss": margins_loss.item(),
        "pre_att_loss": pre_att_loss.item(),
        "longest_diameter_loss": longest_diameter_loss.item(),
        "longest_perp_diameter_loss": longest_perp_diameter_loss.item()
    }
    return total_loss, loss_dict


class AuxVisionDataset(Dataset):

    def __init__(self, json_path, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform

        self.labels_order = labels_order
        self.abnormality_type_0_index = 0
        self.abnormality_type_1_start_index = 1
        self.abnormality_type_1_end_index = 5
        self.abnormality_type_2_index = 6
        self.abnormality_type_3_index = 7
        self.abnormality_type_4_index = 8
        self.abnormality_type_5_index = 9
        self.abnormality_type_6_index = 10
        self.abnormality_type_7_index = 11
        self.abnormality_type_8_index = 12
        self.abnormality_type_9_index = 13
        self.abnormality_type_10_start_index = 14
        self.abnormality_type_10_end_index = 15
        self.abnormality_type_11_index = 16
        self.abnormality_type_12_index = 17
        self.abnormality_type_13_start_index = 18
        self.abnormality_type_13_end_index = 21
        self.abnormality_type_14_start_index = 22
        self.abnormality_type_14_end_index = 24
        self.abnormality_type_15_start_index = 25
        self.abnormality_type_15_end_index = 27
        self.abnormality_type_index_map = {0: self.abnormality_type_0_index,
                                           1: (self.abnormality_type_1_start_index, self.abnormality_type_1_end_index),
                                           2: self.abnormality_type_2_index,
                                           3: self.abnormality_type_3_index,
                                           4: self.abnormality_type_4_index,
                                           5: self.abnormality_type_5_index,
                                           6: self.abnormality_type_6_index,
                                           7: self.abnormality_type_7_index,
                                           8: self.abnormality_type_8_index,
                                           9: self.abnormality_type_9_index,
                                           10: (self.abnormality_type_10_start_index, self.abnormality_type_10_end_index),
                                           11: self.abnormality_type_11_index,
                                           12: self.abnormality_type_12_index,
                                           13: (self.abnormality_type_13_start_index, self.abnormality_type_13_end_index),
                                           14: (self.abnormality_type_14_start_index, self.abnormality_type_14_end_index),
                                           15: (self.abnormality_type_15_start_index, self.abnormality_type_15_end_index)}
        self.question_keys = [
            "abnormality_type_labels",
            "preexist_labels",
            "location_labels",
            "interval_change_labels",
            "interval_growth_labels",
            "further_investigation_labels",
            "margins_labels",
            "pre_att_labels",
            "longest_diameter_labels",
            "longest_perp_diameter_labels",
        ]

        # Load dictionary
        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "img_files": datum_dict["img_files"],
                "filters": datum_dict["filters"],
                "content_info": datum_dict["content_info"]
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

        # Load img file
        img_npy = np.load(img_file_npy)

        if self.transform is not None:
            img_tensor = self.transform(img_npy)

        ci = data["content_info"]
        abnormality_type_lst = ci["abnormality_type"]
        pre_existing_lst = ci["pre_existing"]
        location_lst = ci["location"]
        interval_change_lst = ci["interval_change"]
        interval_growth_lst = ci["interval_growth"]
        further_investigation_lst = ci["further_investigation"]
        margins_lst = ci["margins"]
        predominant_attenuation_lst = ci["predominant_attenuation"]
        longest_diameter_lst = ci["longest_diameter"]
        longest_perp_diameter_lst = ci["longest_perpendicular_diameter"]

        # make one *independent* row per abnormality type
        label_list = [[] for _ in range(len(self.labels_order))]

        for k, abn_type in enumerate(abnormality_type_lst):
            target_indices = self.abnormality_type_index_map[abn_type]

            # convert single int → tuple for unified handling
            if isinstance(target_indices, int):
                target_indices = (target_indices, target_indices)

            for row_idx in range(target_indices[0], target_indices[1] + 1):
                if not label_list[row_idx]:
                    label_list[row_idx] = [
                        1,
                        pre_existing_lst[k],
                        location_lst[k],
                        interval_change_lst[k],
                        interval_growth_lst[k],
                        further_investigation_lst[k],
                        margins_lst[k],
                        predominant_attenuation_lst[k],
                        longest_diameter_lst[k],
                        longest_perp_diameter_lst[k],
                    ]
                    break

        # fill missing rows with zeros / NaNs
        for r in range(len(label_list)):
            if not label_list[r]:
                label_list[r] = [0, 0, 0, 0, 0, 0, 0, 0, float("nan"), float("nan")]

        label_t = np.array(label_list, dtype=np.float32).T
        label_dict = {k: torch.tensor(label_t[i]) for i, k in enumerate(self.question_keys)}

        return {
            "image": img_tensor,
            "label_dict": label_dict,
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
        for rank, pattern in _RULES:
            if pattern.search(kernel):
                return rank
        return 9  # should never hit because last rule is '.*'

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
        self.labels_order = labels_order
        self.vision_tower = vision_tower
        self.use_cls = use_cls
        if self.use_cls:
            cls_hidden_dim = 768
            non_cls_hidden_dim = 768 * 2048

            self.abnormality_type_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(abnormality_type_map))
            self.location_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(location_map))
            self.margins_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(margins_map))
            self.pre_att_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(pre_att_map))
            self.interval_change_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(interval_change_map))
            self.interval_growth_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order) * len(interval_growth_map))
            self.further_investigation_head = nn.Linear(cls_hidden_dim, len(self.labels_order) * len(further_investigation_map))
            self.preexist_head = nn.Linear(cls_hidden_dim, len(self.labels_order) * len(ab_preexist_map))
            self.longest_diameter_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order))
            self.longest_perpendicular_diameter_head = nn.Linear(non_cls_hidden_dim, len(self.labels_order))
        else:
            hidden_dim = 768 * 2048

            self.abnormality_type_head = nn.Linear(hidden_dim, len(self.labels_order) * len(abnormality_type_map))
            self.location_head = nn.Linear(hidden_dim, len(self.labels_order) * len(location_map))
            self.margins_head = nn.Linear(hidden_dim, len(self.labels_order) * len(margins_map))
            self.pre_att_head = nn.Linear(hidden_dim, len(self.labels_order) * len(pre_att_map))
            self.interval_change_head = nn.Linear(hidden_dim, len(self.labels_order) * len(interval_change_map))
            self.interval_growth_head = nn.Linear(hidden_dim, len(self.labels_order) * len(interval_growth_map))
            self.further_investigation_head = nn.Linear(hidden_dim, len(self.labels_order) * len(further_investigation_map))
            self.preexist_head = nn.Linear(hidden_dim, len(self.labels_order) * len(ab_preexist_map))
            self.longest_diameter_head = nn.Linear(hidden_dim, len(self.labels_order))
            self.longest_perpendicular_diameter_head = nn.Linear(hidden_dim, len(self.labels_order))

    def forward(self, image):
        B = image.size(0)

        # Extract features
        feats = self.vision_tower.forward(image)

        if self.use_cls:
            # collect cls and non-cls features
            cls_feats = feats[:, 0]
            cls_feats = cls_feats.view(B, -1)
            non_cls_feats = feats[:, 1:]
            non_cls_feats = non_cls_feats.view(B, -1)
            # model features
            abnormality_type_feats = non_cls_feats
            location_feats = non_cls_feats
            margins_feats = non_cls_feats
            pre_att_feats = non_cls_feats
            interval_change_feats = non_cls_feats
            interval_growth_feats = non_cls_feats
            further_investigation_feats = cls_feats
            preexist_feats = cls_feats
            longest_diameter_feats = non_cls_feats
            longest_perpendicular_diameter_feats = non_cls_feats
        else:
            feats = feats.view(B, -1)
            # model features
            abnormality_type_feats = feats
            location_feats = feats
            margins_feats = feats
            pre_att_feats = feats
            interval_change_feats = feats
            interval_growth_feats = feats
            further_investigation_feats = feats
            preexist_feats = feats
            longest_diameter_feats = feats
            longest_perpendicular_diameter_feats = feats

        abnormality_type_logits = self.abnormality_type_head(abnormality_type_feats).view(B, len(self.labels_order), len(abnormality_type_map))
        location_logits = (self.location_head(location_feats).view(B, len(self.labels_order), len(location_map)))
        margins_logits = (self.margins_head(margins_feats).view(B, len(self.labels_order), len(margins_map)))
        pre_att_logits = (self.pre_att_head(pre_att_feats).view(B, len(self.labels_order), len(pre_att_map)))
        interval_change_logits = (self.interval_change_head(interval_change_feats).view(B,
                                                                                        len(self.labels_order),
                                                                                        len(interval_change_map)))
        interval_growth_logits = (self.interval_growth_head(interval_growth_feats).view(B,
                                                                                        len(self.labels_order),
                                                                                        len(interval_growth_map)))
        further_investigation_logits = (self.further_investigation_head(further_investigation_feats).view(B,
                                                                                                          len(self.labels_order),
                                                                                                          len(further_investigation_map)))
        preexist_logits = (self.preexist_head(preexist_feats).view(B, len(self.labels_order), len(ab_preexist_map)))
        longest_diameter_reg_logits = self.longest_diameter_head(longest_diameter_feats).view(B, len(self.labels_order))
        longest_perp_diameter_reg_logits = self.longest_perpendicular_diameter_head(longest_perpendicular_diameter_feats).view(B, len(self.labels_order))

        return {
            "abnormality_type_logits": abnormality_type_logits,
            "location_logits": location_logits,
            "margins_logits": margins_logits,
            "pre_att_logits": pre_att_logits,
            "interval_change_logits": interval_change_logits,
            "interval_growth_logits": interval_growth_logits,
            "further_investigation_logits": further_investigation_logits,
            "preexist_logits": preexist_logits,
            "longest_diameter_reg_logits": longest_diameter_reg_logits,
            "longest_perp_diameter_reg_logits": longest_perp_diameter_reg_logits,
        }


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

    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    output_dir: str = "./nlst_vision_aux_output"
    device: str = "cuda"
    tag: str = ""
    use_cls: bool = True


def _masked_acc(pred, tgt):
    mask = torch.isfinite(tgt)
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == tgt[mask]).float().mean().item()

def _masked_mse(pred, tgt):
    mask = torch.isfinite(tgt)
    if mask.sum() == 0:
        return 0.0
    return ((pred[mask] - tgt[mask]) ** 2).mean().item()

@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    agg = collections.defaultdict(float)   # metric accumulators
    cnt = collections.defaultdict(int)

    for batch in loader:
        img   = batch["image"].to(device)
        label = {k: v.to(device) for k, v in batch["label_dict"].items()}
        out   = model(img)

        # categorical heads -------------------------------------------------
        for key_pred, key_tgt in [
            ("abnormality_type_logits",   "abnormality_type_labels"),
            ("preexist_logits",           "preexist_labels"),
            ("location_logits",           "location_labels"),
            ("interval_change_logits",    "interval_change_labels"),
            ("interval_growth_logits",    "interval_growth_labels"),
            ("further_investigation_logits", "further_investigation_labels"),
            ("margins_logits",            "margins_labels"),
            ("pre_att_logits",            "pre_att_labels"),
        ]:
            p = torch.argmax(out[key_pred], -1).flatten()
            t = label[key_tgt].long().flatten()
            agg[key_pred.replace("_logits", "_acc")] += _masked_acc(p, t) * len(p)
            cnt[key_pred.replace("_logits", "_acc")] += len(p)

        # regression heads --------------------------------------------------
        pred_diam  = out["longest_diameter_reg_logits"].flatten()
        tgt_diam   = label["longest_diameter_labels"].flatten()
        agg["longest_diameter_mse"] += _masked_mse(pred_diam, tgt_diam) * len(pred_diam)
        cnt["longest_diameter_mse"] += len(pred_diam)

        pred_perp = out["longest_perp_diameter_reg_logits"].flatten()
        tgt_perp  = label["longest_perp_diameter_labels"].flatten()
        agg["longest_perp_diameter_mse"] += _masked_mse(pred_perp, tgt_perp) * len(pred_perp)
        cnt["longest_perp_diameter_mse"] += len(pred_perp)

    # average over dataset
    return {k: agg[k] / max(1, cnt[k]) for k in agg}


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
    train_file = "/local2/amvepa91/MedTrinity-25M/nlst_train_aux_vqa_delta2True_v3.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/nlst_val_aux_vqa_delta2True_v3.json"
    test_file = "/local2/amvepa91/MedTrinity-25M/nlst_test_aux_vqa_delta2True_v3.json"

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    train_lbls = collect_labels(train_loader, device)
    majority_cls, mean_reg = build_baseline(train_lbls)

    logger.info("Majority / mean computed from train set:")
    logger.info(str({**majority_cls, **mean_reg}))

    logger.info("Baseline on train:")
    logger.info(str(eval_baseline(train_loader, majority_cls, mean_reg, device)))
    logger.info("Baseline on val:")
    logger.info(str(eval_baseline(val_loader, majority_cls, mean_reg, device)))
    logger.info("Baseline on test:")
    logger.info(str(eval_baseline(test_loader, majority_cls, mean_reg, device)))

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------
    # 3) Optimizer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_val_score = -float("inf")  # higher = better
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        comp_sums = collections.defaultdict(float)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            image = batch["image"].to(device)
            label_dict = {k: v.to(device) for k, v in batch["label_dict"].items()}

            optimizer.zero_grad()
            results = model(image)

            loss, loss_dict = compute_aux_loss(**results, **label_dict,
                                               num_labels=len(labels_order))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                comp_sums[k] += v

        avg_train_loss = total_loss / len(train_loader)
        comp_means = {k: v / len(train_loader) for k, v in comp_sums.items()}

        logger.info(f"[Epoch {epoch + 1}] Train loss = {avg_train_loss:.5f} " +
                    " ".join([f"{k}={v:.4f}" for k, v in comp_means.items()]))
        model.eval()
        val_total_loss = 0.0
        val_comp_sums = collections.defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                image = batch["image"].to(device)
                label_dict = {k: v.to(device) for k, v in batch["label_dict"].items()}

                outputs = model(image)
                v_loss, v_loss_dict = compute_aux_loss(**outputs, **label_dict,
                                                       num_labels=len(labels_order))

                val_total_loss += v_loss.item()
                for k, v in v_loss_dict.items():
                    val_comp_sums[k] += v

        avg_val_loss = val_total_loss / len(val_loader)
        val_metrics = evaluate(val_loader, model, device)
        val_comp_means = {k: v / len(val_loader) for k, v in val_comp_sums.items()}

        logger.info(f"[Epoch {epoch + 1}]  Val loss = {avg_val_loss:.5f} " +
                    " ".join([f"{k}={v:.4f}" for k, v in val_comp_means.items()])
                    + " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

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
    test_comp_sums = collections.defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            image = batch["image"].to(device)
            label_dict = {k: v.to(device) for k, v in batch["label_dict"].items()}

            outputs = model(image)
            t_loss, t_loss_dict = compute_aux_loss(**outputs, **label_dict,
                                                   num_labels=len(labels_order))

            test_total_loss += t_loss.item()
            for k, v in t_loss_dict.items():
                test_comp_sums[k] += v

    avg_test_loss = test_total_loss / len(test_loader)
    test_comp_means = {k: v / len(test_loader) for k, v in test_comp_sums.items()}
    test_metrics = evaluate(test_loader, model, device)

    logger.info(f"Best-val model Test loss = {avg_test_loss:.5f} " +
                " ".join([f"{k}={v:.4f}" for k, v in test_comp_means.items()])
                + " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))


if __name__ == "__main__":
    main()
