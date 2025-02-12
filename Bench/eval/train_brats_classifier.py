import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json

from transformers import HfArgumentParser
from dataclasses import dataclass, field

import monai.transforms as mtf

from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM


@dataclass
class VisionTrainingArguments:
    """
    Minimal training arguments for the vision classifier.
    Feel free to expand with any arguments you need.
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
    pretrain_mllm: str = field(
        default=None,
        metadata={"help": "Path to a pretrained MLLM weights to load into the model (optional)."}
    )

    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the entire vision tower during training."}
    )

    num_labels: int = field(
        default=5,
        metadata={"help": "Number of labels for multi-label classification."}
    )

    # Basic training settings
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    output_dir: str = "./vision_classifier_output"
    device: str = "cuda"



class MultiLabelVisionDataset(Dataset):
    """
    A multi-label vision-only dataset that:
      1) Reads the same JSON structure used by VQABratsDataset.
      2) Groups entries by 'volume_file_dir'.
      3) **Only considers 'content_type' == 'area'** to determine the presence/absence of each label:
         - If answer != 'None', label is present
         - If answer == 'None', label is absent
      4) Creates a multi-hot label vector for each volume by aggregating all present labels.
      5) Loads 4 modalities per volume into shape [4, C, D, H, W].
      6) Returns {'image': Tensor, 'labels': multi-hot Tensor}.
    """

    def __init__(self, data_file, mode="train"):
        """
        :param args:  Typically your DataArguments or config with fields:
                      - vqa_data_train_path / val_path / test_path
                      - ...
        :param mode:  'train', 'validation', or 'test'
        """
        super().__init__()
        self.mode = mode

        # ------------------------------------------------------
        # Define transforms (same as your VQABrats code)
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
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform
        else:
            raise ValueError(f"Unknown mode {mode}.")
        self.data_file = data_file

        # Read & group JSON data so each volume_file_dir has:
        #   1) volume_non_seg_files: { "t1c":..., "t1n":..., "t2f":..., "t2w":... }
        #   2) a set (or dict) indicating which labels are present
        self.data_by_vol = self._read_and_group_data(self.data_file)

        # Build a global sorted list of all possible labels.
        # We'll create a label->index mapping for multi-hot vectors.
        all_labels = set()
        for vol_dir, vol_info in self.data_by_vol.items():
            all_labels |= vol_info["present_labels"]  # union of sets
        self.all_labels_sorted = sorted(list(all_labels))
        self.label2id = {lbl: i for i, lbl in enumerate(self.all_labels_sorted)}

        # Flatten into a list of volumes for __getitem__ indexing
        self.samples = []
        for vol_dir, vol_info in self.data_by_vol.items():
            self.samples.append({
                "volume_file_dir": vol_dir,
                "volume_non_seg_files": vol_info["volume_non_seg_files"],
                # Convert set of labels into a multi-hot vector
                "label_vec": self._labels_to_multihot(vol_info["present_labels"]),
            })

    def _labels_to_multihot(self, present_labels):
        """
        Convert a set of label names to a multi-hot vector (FloatTensor).
        """
        vec = torch.zeros(len(self.all_labels_sorted), dtype=torch.float)
        for lbl in present_labels:
            idx = self.label2id[lbl]
            vec[idx] = 1.0
        return vec

    def _read_and_group_data(self, json_path):
        """
        1) Load the JSON (like VQABratsDataset).
        2) Group by volume_file_dir.
        3) For each volume_file_dir:
             - Store one volume_non_seg_files (from the first matching entry).
             - Collect which labels are present (where content_type=="area" and answer!="None").
        """
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        grouped = {}
        for entry in raw_data:
            vol_dir = entry["volume_file_dir"]
            if vol_dir not in grouped:
                grouped[vol_dir] = {
                    "volume_non_seg_files": entry["volume_non_seg_files"],
                    "present_labels": set(),   # We'll fill this in below
                }

            # We only care about content_type=="area".
            if entry["content_type"] == "area":
                label = entry["label_name"]  # e.g. "tumor", "necrosis", "edema"
                answer_str = str(entry["answer"])  # might be "None", "12.3", "56", etc.
                if answer_str.lower() != "none":
                    # Mark this label as present
                    grouped[vol_dir]["present_labels"].add(label)

        return grouped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary:
          {
            "image": FloatTensor, shape [4, C, D, H, W]
            "labels": FloatTensor, shape [num_labels] (multi-hot)
            "volume_file_dir": ...
          }
        """
        sample = self.samples[idx]
        vol_dir = sample["volume_file_dir"]
        label_vec = sample["label_vec"].clone()  # multi-hot

        # Load each modality to build [4, C, D, H, W]
        modalities = ["t1c", "t1n", "t2f", "t2w"]
        modality_tensors = []
        for modality in modalities:
            npy_path = sample["volume_non_seg_files"][modality]
            npy_path = self.convert_file_path_to_npy(npy_path)  # same logic as VQABrats
            vol = np.load(npy_path)      # shape e.g. (1, 32, 256, 256) or (C, D, H, W)
            vol = self.transform(vol)    # apply MONAI transforms
            modality_tensors.append(vol)

        return_dict = {modality: image for modality, image in zip(modalities, modality_tensors)}
        return_dict["labels"] = label_vec
        return_dict["volume_file_dir"] = vol_dir
        return return_dict

    def convert_file_path_to_npy(self, image_abs_path):
        """
        Same logic from VQABratsDataset to map an original path to a '.npy' location.
        Adjust if your data layout differs.
        """
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"

        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path


# ------------------------------------------------------------------------
# Vision classifier that wraps the vision tower + a custom classifier head
# ------------------------------------------------------------------------
class VisionMultiLabelClassifier(nn.Module):
    def __init__(self, vision_tower: nn.Module, num_labels: int):
        """
        :param vision_tower: The extracted vision model (e.g. `model.get_model().vision_tower`).
        :param num_labels: Number of labels for multi-label classification.
        """
        super().__init__()
        self.vision_tower = vision_tower

        # Inspect the dimension of the output from the vision tower.
        # Suppose we get a feature vector of dimension 'hidden_dim'.
        # If uncertain, print out shapes in a debug run or check the tower definition.
        hidden_dim = 768  # or whatever your vision model produces
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, mod1, mod2, mod3, mod4, labels=None):
        feats1 = self.vision_tower(mod1)
        feats2 = self.vision_tower(mod2)
        feats3 = self.vision_tower(mod3)
        feats4 = self.vision_tower(mod4)
        feats = torch.cat([feats1, feats2, feats3, feats4], dim=1)
        logits = self.classifier(feats)

        if labels is not None:
            # For multi-label classification, we typically use BCEWithLogitsLoss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits


# ------------------------------------------------------------------------
# Main training script
# ------------------------------------------------------------------------
def main():
    parser = HfArgumentParser(VisionTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_v2.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_v2.json"
    test_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json"

    # --------------------------------------------------------------------
    # 1) Load the pre-trained MLLM with a vision tower
    # --------------------------------------------------------------------
    if 'llama' in args.model_type.lower():
        base_model = LamedLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    elif 'phi3' in args.model_type.lower():
        base_model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}. Supported: ['llama2', 'phi3']")

    vision_tower = base_model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError(
            "No vision tower found in the loaded model. Ensure `vision_tower` is correctly specified."
        )

    # Optionally freeze the entire vision tower
    if args.freeze_vision_tower:
        for param in vision_tower.parameters():
            param.requires_grad = False
        print("Vision tower is frozen.")

    # Create our classification model
    model = VisionMultiLabelClassifier(vision_tower=vision_tower, num_labels=args.num_labels).to(device)


    train_dataset = MultiLabelVisionDataset(data_file=train_file, mode="train")
    val_dataset = MultiLabelVisionDataset(data_file=val_file, mode="validation")
    test_dataset = MultiLabelVisionDataset(data_file=test_file, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --------------------------------------------------------------------
    # 3) Set up the optimizer
    # --------------------------------------------------------------------
    # Include only the model parameters that `requires_grad = True`
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate)

    # --------------------------------------------------------------------
    # 4) Training Loop
    # --------------------------------------------------------------------
    best_val_loss = float('inf')
    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss, logits = model(images, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # ------------------------------
        #  Validation
        # ------------------------------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                loss, logits = model(images, labels=labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Validation Loss: {val_loss:.4f}")
        model.train()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best val loss. Model saved to {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
