# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import random

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out





# class ViT3DTower(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.select_layer = config.vision_select_layer
#         self.select_feature = config.vision_select_feature

#         self.vision_tower = ViT(
#             in_channels=self.config.image_channel,
#             img_size=self.config.image_size, # manually set to 1/4 of the original size to consider all modalities
#             patch_size=self.config.patch_size,
#             pos_embed="perceptron",
#             spatial_dims=len(self.config.patch_size),
#             classification=True,
#         )

#     def forward(self, images):
#         last_feature, hidden_states = self.vision_tower(images)
#         if self.select_layer == -1:
#             image_features = last_feature
#         elif self.select_layer < -1:
#             image_features = hidden_states[self.select_feature]
#         else:
#             raise ValueError(f'Unexpected select layer: {self.select_layer}')

#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')

#         return image_features

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def hidden_size(self):
#         return self.vision_tower.hidden_size




class VisionReconstructionDecoder(nn.Module):
    """Decoder for reconstruction self-supervised learning task."""
    def __init__(self, hidden_size=768, patch_size=(16, 16, 16), image_size=(128, 128, 128)):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Convert lists to tuples if needed
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        if isinstance(image_size, list):
            image_size = tuple(image_size)
            
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size, image_size)
        
        # Now extract values as integers for calculation
        p_d, p_h, p_w = self.patch_size
        i_d, i_h, i_w = self.image_size
        
        # Calculate number of patches for each dimension
        self.num_patches_d = i_d // p_d
        self.num_patches_h = i_h // p_h
        self.num_patches_w = i_w // p_w
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        
        # Calculate patch volume
        self.patch_volume = p_d * p_h * p_w
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, self.patch_volume)  # Reconstruct patch
        )
    
    def forward(self, x):
        """
        Args:
            x: Token embeddings from ViT [B, num_patches, hidden_size]
        Returns:
            Reconstructed patches [B, num_patches, patch_volume]
        """
        # x has shape [B, num_patches, hidden_size]
        B = x.shape[0]
        reconstructed_patches = self.decoder(x)  # [B, num_patches, patch_volume]
        return reconstructed_patches


class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        # Main ViT encoder
        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )
        
        # Reconstruction decoder
        self.decoder = VisionReconstructionDecoder(
            hidden_size=768,  # Match hidden_size from ViT
            patch_size=self.config.patch_size,
            image_size=self.config.image_size
        )
        
        # Parameters for masking
        self.mask_ratio = getattr(config, 'mask_ratio', 0.3)  # Default to 30% masking

    def apply_random_mask(self, features, mask_ratio=None):
        """Apply random masking to features
        Args:
            features: [B, N, D] where N is num_patches+1 ([CLS] token included)
            mask_ratio: Ratio of patches to mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        B, N, D = features.shape
        
        # Don't mask the cls token (first token)
        cls_token = features[:, 0:1, :]
        patch_tokens = features[:, 1:, :]
        
        L = patch_tokens.shape[1]  # Number of patches
        num_mask = int(L * mask_ratio)
        
        # Create random mask indices for each batch
        mask_indices = []
        for _ in range(B):
            # Random indices of patches to mask
            mask_idx = random.sample(range(L), num_mask)
            mask_indices.append(mask_idx)
            
        # Create mask tensor (1 = keep, 0 = mask)
        mask = torch.ones(B, L, device=features.device)
        for b in range(B):
            mask[b, mask_indices[b]] = 0
            
        # Apply mask: replace masked tokens with zeros
        masked_patch_tokens = patch_tokens * mask.unsqueeze(-1)
        
        # Reconstruct original shape with cls token
        masked_features = torch.cat([cls_token, masked_patch_tokens], dim=1)
        
        return masked_features, mask, mask_indices

    def forward(self, images, apply_mask=False):
        # Get features from vision tower
        last_feature, hidden_states = self.vision_tower(images)
        
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')
            
        if apply_mask:
            # Apply masking for reconstruction task
            masked_features, mask, mask_indices = self.apply_random_mask(image_features)
            
            # Extract patch tokens (excluding cls token)
            patch_tokens = masked_features[:, 1:, :]
            
            # Generate reconstructed patches
            reconstructed_patches = self.decoder(patch_tokens)
            
            return image_features, masked_features, reconstructed_patches, mask, mask_indices
        else:
            # Normal forward pass without reconstruction
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
                
            return image_features



    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size
