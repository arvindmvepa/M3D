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
import torch.nn.functional as F

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
        print('self.patch_volume ', self.patch_volume )
        print('p_d, p_h, p_w', p_d, p_h, p_w)
        
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
        self.patch_size = self.config.patch_size
        
        # Calculate important dimensions for patching
        pd, ph, pw = self.patch_size
        i_d, i_h, i_w = self.config.image_size
        self.num_patches_d = i_d // pd
        self.num_patches_h = i_h // ph
        self.num_patches_w = i_w // pw
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w

        # Main ViT encoder
        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )
        # Parameters for masking
        self.mask_ratio = getattr(config, 'mask_ratio', 0.3)  # Default to 30% masking
        
        if hasattr(config, 'modality_weights'):
            self.modality_weights = self.config.modality_weights
        # Initialize decoders based on requirements
        self.num_modalities = getattr(config, 'num_modalities', 4)
        
        # Create self-reconstruction decoders for each modality
        self.decoders = nn.ModuleList([
            VisionReconstructionDecoder(
                hidden_size=768, 
                patch_size=self.config.patch_size, 
                image_size=self.config.image_size
            )
            for _ in range(self.num_modalities)
        ])
        
        # Create cross-modal decoders based on cross_modality_matrix
        self.cross_decoders = nn.ModuleDict()
        
        if hasattr(config, 'cross_modality_matrix'):
            self.cross_modality_matrix = self.config.cross_modality_matrix
            self._create_cross_modal_decoders(self.cross_modality_matrix)
        
    
    def _create_cross_modal_decoders(self, cross_modality_matrix):
        """
        Create cross-modal decoders based on non-zero entries in the cross-modality matrix.
        """
        matrix = cross_modality_matrix.cpu().numpy()
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j and matrix[i, j] > 0:  # Skip self-reconstruction and zero-weighted pairs
                    key = f'{i}_to_{j}'
                    self.cross_decoders[key] = VisionReconstructionDecoder(
                        hidden_size=768,
                        patch_size=self.config.patch_size,
                        image_size=self.config.image_size
                    )
        
        if len(self.cross_decoders) > 0:
            print(f"Created {len(self.cross_decoders)} cross-modal decoders:")
            for key in self.cross_decoders.keys():
                source, target = key.split('_to_')
                print(f"  - Modality {source} → Modality {target}")

    def extract_image_patches(self, images):
        """
        Extract patches from original images
        Args:
            images: [B, C, D, H, W]
        Returns:
            patches: [B, num_patches, C*pd*ph*pw] where num_patches is the total number of patches
        """
        B, C, D, H, W = images.shape
        pd, ph, pw = self.patch_size
        
        # Number of patches in each dimension
        nd, nh, nw = D // pd, H // ph, W // pw
        
        # Reshape to extract patches
        patches = images.unfold(2, pd, pd).unfold(3, ph, ph).unfold(4, pw, pw)
        patches = patches.contiguous().view(B, C, nd, nh, nw, pd, ph, pw)
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        patches = patches.view(B, nd*nh*nw, C*pd*ph*pw)
        
        return patches


    def mask_input_patches(self, images):
        """
        Mask patches preferentially from center regions of the image
        
        Args:
            images: [B, C, D, H, W]
        Returns:
            masked_images: [B, C, D, H, W] with masked patches
            original_patches: [B, num_patches, patch_volume]
            mask_indices: Indices of masked patches
            patch_positions: Original 3D positions of all patches
        """
        B, C, D, H, W = images.shape
        pd, ph, pw = self.patch_size
        
        # Number of patches in each dimension
        nd, nh, nw = D // pd, H // ph, W // pw
        num_patches = nd * nh * nw
        
        # Extract original patches before masking
        original_patches = self.extract_image_patches(images)
        
        # Create a copy of images to apply masking
        masked_images = images.clone()
        
        # Define center region - typically in brain MRI the center 60% contains most of the tissue
        center_d_start, center_d_end = int(0.2 * nd), int(0.8 * nd)
        center_h_start, center_h_end = int(0.2 * nh), int(0.8 * nh)
        center_w_start, center_w_end = int(0.2 * nw), int(0.8 * nw)
        
        # Create a list of center patch indices
        center_patch_indices = []
        for d_idx in range(center_d_start, center_d_end):
            for h_idx in range(center_h_start, center_h_end):
                for w_idx in range(center_w_start, center_w_end):
                    # Convert 3D position to flat index
                    idx = d_idx * (nh * nw) + h_idx * nw + w_idx
                    center_patch_indices.append(idx)
        
        # Convert to tensor
        center_patch_indices = torch.tensor(center_patch_indices, device=images.device)
        
        # Mask indices for each batch
        mask_indices = []
        for b in range(B):
            # Calculate how many patches to mask
            num_mask = min(int(num_patches * self.mask_ratio), len(center_patch_indices))
            
            if len(center_patch_indices) > 0:
                # Randomly select indices from center patches
                perm = torch.randperm(len(center_patch_indices), device=images.device)
                mask_idx = center_patch_indices[perm[:num_mask]]
            else:
                # Fall back to random selection if no center patches
                perm = torch.randperm(num_patches, device=images.device)
                mask_idx = perm[:num_mask]
            
            mask_indices.append(mask_idx)
            
            # Apply masking to the image
            for idx in mask_idx:
                # Convert flat index to 3D position
                d_idx = idx // (nh * nw)
                hw_idx = idx % (nh * nw)
                h_idx = hw_idx // nw
                w_idx = hw_idx % nw
                
                # Calculate starting position in the original image
                d_start = d_idx * pd
                h_start = h_idx * ph
                w_start = w_idx * pw
                
                # Set the patch to zero (mask it)
                masked_images[b, :, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] = 0.0
        
        # Store patch positions for reference (helps with visualization)
        patch_positions = []
        for idx in range(num_patches):
            d_idx = idx // (nh * nw)
            hw_idx = idx % (nh * nw)
            h_idx = hw_idx // nw
            w_idx = hw_idx % nw
            patch_positions.append((d_idx, h_idx, w_idx))
            
        return masked_images, original_patches, mask_indices, patch_positions

    def forward_mim_multimodal(self, mod1, mod2, mod3, mod4, apply_mask=True):
        """
        Forward method for multimodal Masked Image Modeling
        Args:
            mod1, mod2, mod3, mod4: Input images for different modalities [B, C, D, H, W]
            apply_mask: Whether to apply masking
        Returns:
            Dictionary with features and reconstructions for all modalities
        """
        B = mod1.shape[0]
        
        modalities = [mod1, mod2, mod3, mod4]
        modality_results = []
        
        # Process each modality
        for mod_idx, mod in enumerate(modalities):
            if not apply_mask:
                # Standard forward pass without masking
                features, hidden_states = self.vision_tower(mod)
                modality_results.append({'features': features})
                continue
            
            # 1. Mask random patches in the input image
            masked_images, original_patches, mask_indices, patch_positions = self.mask_input_patches(mod)
            
            # 2. Forward pass with masked images
            features, hidden_states = self.vision_tower(masked_images)
            
            # 3. Extract patch tokens (excluding cls token if present)
            if self.select_feature == 'patch':
                patch_tokens = features[:, 1:]
            elif self.select_feature == 'cls_patch':
                patch_tokens = features[:, 1:]  # Exclude CLS token
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            
            # 4. Reconstruct patches using the appropriate decoder
            reconstructed_patches = self.decoders[mod_idx](patch_tokens)
            
            modality_results.append({
                'features': features,
                'original_patches': original_patches,
                'reconstructed_patches': reconstructed_patches,
                'mask_indices': mask_indices,
                'patch_positions': patch_positions,
                'masked_images': masked_images
            })
        
        # 5. Cross-modal reconstruction based on cross-modal decoders
        cross_reconstructed = {}
        if apply_mask and len(self.cross_decoders) > 0:
            for key, decoder in self.cross_decoders.items():
                source_idx, target_idx = map(int, key.split('_to_'))
                source_features = modality_results[source_idx]['features']
                source_patch_tokens = source_features[:, 1:]  # Exclude CLS token
                cross_reconstructed[key] = decoder(source_patch_tokens)
        
        combined_results = {
            'modality_results': modality_results,
            'cross_reconstructed': cross_reconstructed
        }
        
        return combined_results
    
    def compute_mim_loss(self, original_patches, reconstructed_patches, mask_indices, reduction='mean'):
        """
        Compute MSE loss between original and reconstructed patches (only for masked patches)
        Args:
            original_patches: [B, num_patches, patch_volume]
            reconstructed_patches: [B, num_patches, patch_volume]
            mask_indices: List of indices of masked patches for each batch
            reduction: 'mean' or 'sum'
        Returns:
            loss: MSE loss for masked patches
        """
        B = original_patches.shape[0]
        loss = 0.0
        
        for b in range(B):
            # Extract masked patches
            masked_idx = mask_indices[b]
            if len(masked_idx) > 0:
                # Get original and reconstructed patches
                orig = original_patches[b, masked_idx]
                recon = reconstructed_patches[b, masked_idx]
                
                # Compute MSE loss
                batch_loss = F.mse_loss(recon, orig, reduction=reduction)
                loss += batch_loss
        
        if reduction == 'mean' and B > 0:
            loss /= B
            
        return loss
    
    def compute_multimodal_mim_loss(self, 
                                    original_patches_list,  # List of [B, num_patches, patch_volume] for each modality
                                    reconstructed_patches_list,  # List of [B, num_patches, patch_volume] for each modality
                                    cross_reconstructed_patches_dict, 
                                    mask_indices_list,  # List of mask indices for each modality
                                    reduction='mean'):
        """
        Compute loss for multimodal MIM (self-reconstruction + cross-modal reconstruction)
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Self-reconstruction loss
        self_recon_loss = 0.0
        mod_losses = []
        
        for mod_idx, (orig_patches, reconstructed_patches, mask_indices) in enumerate(zip(original_patches_list, reconstructed_patches_list, mask_indices_list)):
            mod_loss = self.compute_mim_loss(orig_patches, reconstructed_patches, mask_indices)
            weighted_mod_loss = mod_loss * self.modality_weights[mod_idx]
            self_recon_loss += weighted_mod_loss
            mod_losses.append(mod_loss.item())
            
        # Cross-modal reconstruction loss
        cross_recon_loss = 0.0
        cross_losses = []
        
        # We're only doing cross-modal reconstruction from T2f (mod index 2) to other modalities
        source_idx = 2  # T2f 
        for target_idx in [0, 1, 3]:  # T1c, T1n, T2w
            key = f'{source_idx}_to_{target_idx}'
            if key in cross_reconstructed_patches_dict:
                cross_loss = self.compute_mim_loss(
                    original_patches_list[target_idx],  # Target modality original
                    cross_reconstructed_patches_dict[key],  # Source → Target reconstruction
                    mask_indices_list[source_idx]  # Masked indices from source modality
                )
                weighted_cross_loss = cross_loss * self.cross_modality_matrix[source_idx, target_idx]
                cross_recon_loss += weighted_cross_loss
                cross_losses.append((source_idx, target_idx, cross_loss.item()))
            
        # Total loss
        total_loss = self_recon_loss + cross_recon_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'self_reconstruction': self_recon_loss.item(),
            'cross_reconstruction': cross_recon_loss.item() if isinstance(cross_recon_loss, torch.Tensor) else cross_recon_loss
        }
        
        # Add individual modality losses for monitoring
        for mod_idx, loss in enumerate(mod_losses):
            loss_dict[f'mod{mod_idx}_recon_loss'] = loss
            
        # Add cross-modal reconstruction losses
        for source_idx, target_idx, loss in cross_losses:
            loss_dict[f'cross_{source_idx}_to_{target_idx}_loss'] = loss
        
        return total_loss, loss_dict


    
    def compute_cross_modality_reconstruction_loss(self, 
        original_patches_list,  # List of [B, num_patches, patch_volume] for each modality
        reconstructed_patches_list,  # List of [B, num_patches, patch_volume] for each modality
        cross_reconstructed_patches_dict, 
        mask_indices_list,  # List of mask indices for each modality
        reduction='mean'
    ):      
        B = original_patches_list[0].shape[0]
        num_modalities = len(original_patches_list)
        
        # Dictionary to store individual loss components
        loss_dict = {
            'self_reconstruction': 0.0,
            'cross_reconstruction': 0.0
        }
        
        # Individual modality losses (self-reconstruction)
        individual_losses = []
        for mod_idx in range(num_modalities):
            original = original_patches_list[mod_idx]
            reconstructed = reconstructed_patches_list[mod_idx]
            mask_indices = mask_indices_list[mod_idx]
            
            mod_loss = 0
            num_batches = 0
            for b in range(B):
                # Extract only the masked patches that need to be reconstructed
                if b < len(mask_indices):  # Check if this batch has mask indices
                    masked_idx = mask_indices[b]
                    if len(masked_idx) > 0:  # Only compute if there are masked patches
                        target = original[b, masked_idx]
                        pred = reconstructed[b, masked_idx]
                        
                        # Compute MSE loss for this batch
                        batch_loss = F.mse_loss(pred, target, reduction=reduction)
                        mod_loss += batch_loss
                        num_batches += 1
                
            # Average over batches if using mean reduction
            if reduction == 'mean' and num_batches > 0:
                mod_loss /= num_batches
                
            individual_losses.append(mod_loss)
            # Apply modality weight
            weighted_mod_loss = mod_loss * self.modality_weights[mod_idx]
            loss_dict['self_reconstruction'] += weighted_mod_loss
        
        # Cross-modality reconstruction (predicting one modality from another)
        cross_losses = []
        for key, cross_reconstructed in cross_reconstructed_patches_dict.items():
            # Parse source and target indices from key (format: "source_idx_to_target_idx")
            source_idx, target_idx = map(int, key.split('_to_'))
            
            # Only compute if there's a non-zero weight in the cross-modality matrix
            if self.cross_modality_matrix[source_idx, target_idx] > 0:
                # We're using masked positions from source modality but reconstructing target modality content
                target_original = original_patches_list[target_idx]
                source_mask_indices = mask_indices_list[source_idx]
                
                cross_loss = 0
                num_batches = 0
                for b in range(B):
                    if b < len(source_mask_indices):  # Check if this batch has mask indices
                        masked_idx = source_mask_indices[b]
                        if len(masked_idx) > 0:  # Only compute if there are masked patches
                            source_pred = cross_reconstructed[b, masked_idx]
                            target_ground_truth = target_original[b, masked_idx]
                            
                            # Compute MSE loss for this cross-modal reconstruction
                            batch_cross_loss = F.mse_loss(source_pred, target_ground_truth, reduction=reduction)
                            cross_loss += batch_cross_loss
                            num_batches += 1
                    
                # Average over batches if using mean reduction
                if reduction == 'mean' and num_batches > 0:
                    cross_loss /= num_batches
                    
                weighted_cross_loss = cross_loss * self.cross_modality_matrix[source_idx, target_idx]
                cross_losses.append((source_idx, target_idx, cross_loss.item()))
                loss_dict['cross_reconstruction'] += weighted_cross_loss
        
        # Calculate final loss
        total_loss = loss_dict['self_reconstruction'] + loss_dict['cross_reconstruction']
        
        # Convert tensor values to float for the loss dictionary
        if isinstance(loss_dict['self_reconstruction'], torch.Tensor):
            loss_dict['self_reconstruction'] = loss_dict['self_reconstruction'].item()
        if isinstance(loss_dict['cross_reconstruction'], torch.Tensor):
            loss_dict['cross_reconstruction'] = loss_dict['cross_reconstruction'].item()
        
        # Add individual modality losses to the dictionary for monitoring
        for i, loss in enumerate(individual_losses):
            loss_dict[f'mod{i+1}_loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss
            
        # Add individual cross-modal losses for monitoring
        for source_idx, target_idx, loss in cross_losses:
            loss_dict[f'cross_{source_idx}_to_{target_idx}_loss'] = loss
            
        return total_loss, loss_dict
    
    def forward(self, images, apply_mask=False, multimodal=False, mod2=None, mod3=None, mod4=None):
        """Unified forward method with support for both single and multiple modalities
        
        Args:
            images: Input image(s) [B, C, D, H, W] (first modality or single modality)
            apply_mask: Whether to apply masking for reconstruction
            multimodal: Whether this is a multimodal forward pass
            mod2, mod3, mod4: Additional modalities when multimodal=True
            
        Returns:
            For single modality with apply_mask=False:
                Features tensor [B, N, D]
            For single modality with apply_mask=True:
                Dictionary with features, masked_features, etc.
            For multimodal with apply_mask=True:
                Dictionary with features, reconstructions, etc. for all modalities
        """
        # Single modality case
        if not multimodal:
            return self.process_single_modality(images, 0, apply_mask)
        
        # Multimodal case - requires all modalities
        # Collect all modalities that are provided
        modalities = [images]  # images is always mod1
        if mod2 is not None:
            modalities.append(mod2)
        if mod3 is not None:
            modalities.append(mod3)
        if mod4 is not None:
            modalities.append(mod4)
        
        # Check if we have the expected number of modalities
        expected_modalities = getattr(self, 'num_modalities', 4)
        if len(modalities) != expected_modalities:
            raise ValueError(f"Expected {expected_modalities} modalities, but got {len(modalities)}")
        
        # Process each modality with its corresponding decoder
        if apply_mask:
            # Process each modality with its corresponding index
            modality_results = [self.process_single_modality(mod, mod_idx, True) 
                               for mod_idx, mod in enumerate(modalities)]
            
            # Handle cross-modal reconstructions dynamically based on cross_decoders
            cross_recon = {}
            if hasattr(self, 'cross_decoders') and self.cross_decoders:
                for key, decoder in self.cross_decoders.items():
                    source_idx, target_idx = map(int, key.split('_to_'))
                    
                    # Check if source modality exists and has masked features
                    if source_idx < len(modality_results) and 'masked_features' in modality_results[source_idx]:
                        source_patch_tokens = modality_results[source_idx]['masked_features'][:, 1:, :]
                        cross_recon[key] = decoder(source_patch_tokens)
            
            results = {
                'modality_results': modality_results,
                'cross_reconstructed': cross_recon
            }
        else:
            # Standard forward pass without masking
            features_list = [self.process_single_modality(mod, mod_idx, False)
                            for mod_idx, mod in enumerate(modalities)]
            
            results = {
                'features': features_list
            }
            
        return results
    
    def process_single_modality(self, image, modality_idx=0, apply_mask=False):
        """Process a single modality
        Args:
            image: Input image [B, C, D, H, W]
            modality_idx: Index of the modality (0-3 for t1c, t1n, t2f, t2w)
            apply_mask: Whether to apply masking
        Returns:
            dict containing all needed outputs including features, patches, etc.
        """
        # Extract raw image patches first (for reconstruction targets)
        original_patches = None
        masked_images = None
        mask_indices = None
        patch_positions = None
        
        if apply_mask:
            # Apply masking directly to the input images
            masked_images, original_patches, mask_indices, patch_positions = self.mask_input_patches(image)
            # Forward pass with masked images
            image_features, hidden_states = self.vision_tower(masked_images)
        else:
            # Standard forward pass without masking
            image_features, hidden_states = self.vision_tower(image)
        
        if self.select_layer == -1:
            selected_features = image_features
        elif self.select_layer < -1:
            selected_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')
            
        # Apply reconstruction if needed
        reconstructed_patches = None
        if apply_mask:
            # Extract patch tokens (excluding cls token)
            patch_tokens = selected_features[:, 1:, :]
            
            # Use correct decoder for this modality
            reconstructed_patches = self.decoders[modality_idx](patch_tokens)
            
            return {
                'features': selected_features,
                'masked_features': selected_features,
                'original_patches': original_patches,
                'reconstructed_patches': reconstructed_patches,
                'mask_indices': mask_indices,
                'patch_positions': patch_positions,
                'masked_images': masked_images
            }
        else:
            # Normal forward pass without reconstruction
            if self.select_feature == 'patch':
                selected_features = selected_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                selected_features = selected_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
                
            return selected_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size