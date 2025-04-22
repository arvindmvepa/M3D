import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import json
import random
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Add the base path to allow imports
base_path = "/local2/jrgan/NephrologyKG/M3D"
sys.path.append(base_path)
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM

# Import dataset and model classes from your training code
from train_brats_aux_tasks_recon import AuxVisionDataset, VisionAuxClassifier


def visualize_single_patch_reconstruction(original_patches, reconstructed_patches, mask_indices, patch_positions, 
                                        patch_size=(4, 16, 16), batch_idx=0, patch_idx=0, modality_name=""):
    """
    Visualize a single patch reconstruction with detailed views
    
    Args:
        original_patches: Original patches [B, num_patches, patch_volume]
        reconstructed_patches: Reconstructed patches [B, num_patches, patch_volume]
        mask_indices: List of masked indices for each batch
        patch_positions: List of 3D positions of patches
        patch_size: Size of each patch (D, H, W)
        batch_idx: Which batch to visualize
        patch_idx: Which patch to visualize from the masked patches
        modality_name: Name of the modality
    
    Returns:
        fig: Matplotlib figure object
    """
    pd, ph, pw = patch_size
    patch_volume = pd * ph * pw
    
    # Get the masked indices for this batch
    masked_idx = mask_indices[batch_idx]
    
    if patch_idx >= len(masked_idx):
        print(f"Patch index {patch_idx} exceeds number of masked patches ({len(masked_idx)})")
        return None
    
    # Get the global patch index (from the full set of patches)
    global_patch_idx = masked_idx[patch_idx]
    
    # Get original and reconstructed patch
    orig_patch = original_patches[batch_idx, global_patch_idx].reshape(pd, ph, pw)
    recon_patch = reconstructed_patches[batch_idx, global_patch_idx].reshape(pd, ph, pw)
    
    # Create a figure with multiple views
    fig = plt.figure(figsize=(20, 8))
    
    # Show all depth slices
    for d in range(pd):
        # Original patch slice
        ax1 = fig.add_subplot(2, pd, d+1)
        im1 = ax1.imshow(orig_patch[d], cmap='gray')
        ax1.set_title(f'Original Slice {d+1}/{pd}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Reconstructed patch slice
        ax2 = fig.add_subplot(2, pd, pd+d+1)
        im2 = ax2.imshow(recon_patch[d], cmap='gray')
        ax2.set_title(f'Reconstructed Slice {d+1}/{pd}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Calculate metrics
    mse = F.mse_loss(torch.from_numpy(orig_patch), torch.from_numpy(recon_patch)).item()
    max_signal = np.max(orig_patch)
    psnr = 20 * np.log10(max_signal / np.sqrt(mse)) if mse > 1e-10 else 100
    
    # Add patch position info
    patch_pos = patch_positions[global_patch_idx]
    fig.suptitle(f'{modality_name} - Patch at Position: {patch_pos}\nMSE: {mse:.6f}, PSNR: {psnr:.2f} dB', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    return fig, {'mse': mse, 'psnr': psnr, 'position': patch_pos}


def visualize_cross_modal_single_patch(source_patches, target_patches, cross_recon_patches, 
                                     mask_indices, patch_positions, patch_size=(4, 16, 16), 
                                     batch_idx=0, patch_idx=0, source_name="", target_name=""):
    """
    Visualize a single patch from cross-modal reconstruction
    
    Args:
        source_patches: Source modality patches [B, num_patches, patch_volume]
        target_patches: Target modality original patches [B, num_patches, patch_volume]
        cross_recon_patches: Cross-reconstructed patches [B, num_patches, patch_volume]
        mask_indices: List of masked indices for each batch
        patch_positions: List of 3D positions of patches
        patch_size: Size of each patch (D, H, W)
        batch_idx: Which batch to visualize
        patch_idx: Which patch to visualize from the masked patches
        source_name: Name of source modality
        target_name: Name of target modality
    
    Returns:
        fig: Matplotlib figure object
    """
    pd, ph, pw = patch_size
    
    # Get the masked indices for the source modality
    source_masked_idx = mask_indices[batch_idx]
    
    if patch_idx >= len(source_masked_idx):
        print(f"Patch index {patch_idx} exceeds number of masked patches ({len(source_masked_idx)})")
        return None
    
    # Get the global patch index
    global_patch_idx = source_masked_idx[patch_idx]
    
    # Get patches
    source_patch = source_patches[batch_idx, global_patch_idx].reshape(pd, ph, pw)
    target_patch = target_patches[batch_idx, global_patch_idx].reshape(pd, ph, pw)
    cross_recon_patch = cross_recon_patches[batch_idx, patch_idx].reshape(pd, ph, pw)
    
    # Create a figure with all views
    fig = plt.figure(figsize=(20, 10))
    
    # Show all depth slices
    for d in range(pd):
        # Source patch slice
        ax1 = fig.add_subplot(3, pd, d+1)
        im1 = ax1.imshow(source_patch[d], cmap='gray')
        ax1.set_title(f'{source_name} Slice {d+1}/{pd}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Target patch slice (original)
        ax2 = fig.add_subplot(3, pd, pd+d+1)
        im2 = ax2.imshow(target_patch[d], cmap='gray')
        ax2.set_title(f'{target_name} Original Slice {d+1}/{pd}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Cross-reconstructed patch slice
        ax3 = fig.add_subplot(3, pd, 2*pd+d+1)
        im3 = ax3.imshow(cross_recon_patch[d], cmap='gray')
        ax3.set_title(f'{source_name}→{target_name} Reconstruction Slice {d+1}/{pd}')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Calculate metrics
    mse = F.mse_loss(torch.from_numpy(target_patch), torch.from_numpy(cross_recon_patch)).item()
    max_signal = np.max(target_patch)
    psnr = 20 * np.log10(max_signal / np.sqrt(mse)) if mse > 1e-10 else 100
    
    # Add patch position info
    patch_pos = patch_positions[global_patch_idx]
    fig.suptitle(f'Cross-Modal: {source_name} → {target_name} - Patch at Position: {patch_pos}\nMSE: {mse:.6f}, PSNR: {psnr:.2f} dB', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    return fig, {'mse': mse, 'psnr': psnr, 'position': patch_pos}


def visualize_multiple_patches(modality_results, cross_reconstructed, modality_names=["T1c", "T1n", "T2f", "T2w"],
                             batch_idx=0, num_patches_to_show=5, patch_size=(4, 16, 16), save_dir="patch_visualizations"):
    """
    Visualize multiple patches for both self-reconstruction and cross-modal reconstruction
    
    Args:
        modality_results: Results from forward_mim_multimodal
        cross_reconstructed: Cross-reconstruction results dictionary
        modality_names: Names of the modalities
        batch_idx: Which batch to visualize
        num_patches_to_show: Number of patches to show for each type
        patch_size: Size of each patch
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize self-reconstruction for each modality
    for mod_idx, mod_name in enumerate(modality_names):
        mod_result = modality_results[mod_idx]
        original_patches = mod_result['original_patches'].cpu().numpy()
        reconstructed_patches = mod_result['reconstructed_patches'].cpu().numpy()
        mask_indices = mod_result['mask_indices']
        patch_positions = mod_result['patch_positions']
        
        # Calculate number of patches to show
        num_masked = len(mask_indices[batch_idx])
        num_show = min(num_patches_to_show, num_masked)
        
        print(f"\nVisualizing {num_show} self-reconstruction patches for {mod_name}:")
        print(f"Total masked patches: {num_masked}")
        
        for patch_idx in range(num_show):
            # Random selection for variety
            selected_idx = random.randint(0, num_masked - 1) if num_masked > 1 else 0
            
            fig, metrics = visualize_single_patch_reconstruction(
                original_patches, reconstructed_patches, mask_indices, patch_positions,
                patch_size=patch_size, batch_idx=batch_idx, patch_idx=selected_idx,
                modality_name=mod_name
            )
            
            if fig:
                save_path = os.path.join(save_dir, f"self_recon_{mod_name}_patch{selected_idx}.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Patch {selected_idx} - MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB, Position: {metrics['position']}")
    
    # Visualize cross-modal reconstruction
    if cross_reconstructed:
        for cross_key, cross_patches in cross_reconstructed.items():
            source_idx, target_idx = map(int, cross_key.split('_to_'))
            source_name = modality_names[source_idx]
            target_name = modality_names[target_idx]
            
            source_patches = modality_results[source_idx]['original_patches'].cpu().numpy()
            target_patches = modality_results[target_idx]['original_patches'].cpu().numpy()
            cross_patches_np = cross_patches.cpu().numpy()
            mask_indices = modality_results[source_idx]['mask_indices']
            patch_positions = modality_results[source_idx]['patch_positions']
            
            # Calculate number of patches to show
            num_masked = len(mask_indices[batch_idx])
            num_show = min(num_patches_to_show, num_masked)
            
            print(f"\nVisualizing {num_show} cross-modal patches for {source_name} → {target_name}:")
            
            for patch_idx in range(num_show):
                # Random selection for variety  
                selected_idx = random.randint(0, num_masked - 1) if num_masked > 1 else 0
                
                fig, metrics = visualize_cross_modal_single_patch(
                    source_patches, target_patches, cross_patches_np,
                    mask_indices, patch_positions, patch_size=patch_size,
                    batch_idx=batch_idx, patch_idx=selected_idx,
                    source_name=source_name, target_name=target_name
                )
                
                if fig:
                    save_path = os.path.join(save_dir, f"cross_recon_{source_name}_to_{target_name}_patch{selected_idx}.png")
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Patch {selected_idx} - MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB, Position: {metrics['position']}")



def visualize_full_reconstruction(modality_results, modality_names=["T1c", "T1n", "T2f", "T2w"],
                                batch_idx=0, save_dir="reconstruction_visualizations", device='cuda'):
    """
    Visualize full image reconstruction for each modality
    
    Args:
        modality_results: Results from forward_mim_multimodal
        modality_names: Names of the modalities
        batch_idx: Which batch to visualize
        save_dir: Directory to save visualizations
        device: Device to use
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for mod_idx, mod_name in enumerate(modality_names):
        mod_result = modality_results[mod_idx]
        
        # Get required data
        masked_images = mod_result['masked_images']
        original_patches = mod_result['original_patches']
        reconstructed_patches = mod_result['reconstructed_patches']
        mask_indices = mod_result['mask_indices']
        patch_positions = mod_result['patch_positions']
        
        # Get image sizes
        B, C, D, H, W = masked_images.shape
        patch_size = (4, 16, 16)  # As used in the model
        image_size = (D, H, W)
        
        # Reconstruct full image
        reconstructed_image = reconstruct_full_image(
            masked_images, reconstructed_patches, mask_indices, 
            patch_positions, patch_size, image_size
        )
        
        # For visualization, take center slice
        center_slice = D // 2
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image (before masking)
        original_image = torch.zeros_like(masked_images)
        # We need to get the full original image by placing all patches back
        all_patches = original_patches[batch_idx]  # Get all patches for this batch
        for patch_idx, (d_idx, h_idx, w_idx) in enumerate(patch_positions):
            d_start = d_idx * patch_size[0]
            h_start = h_idx * patch_size[1]
            w_start = w_idx * patch_size[2]
            
            # Reshape from flattened to 3D
            patch = all_patches[patch_idx].reshape(patch_size[0], patch_size[1], patch_size[2])
            original_image[batch_idx, 0, d_start:d_start+patch_size[0], 
                          h_start:h_start+patch_size[1], 
                          w_start:w_start+patch_size[2]] = patch
        
        im0 = axes[0].imshow(original_image[batch_idx, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[0].set_title(f"Original {mod_name}")
        fig.colorbar(im0, ax=axes[0])
        
        # Masked image
        im1 = axes[1].imshow(masked_images[batch_idx, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[1].set_title(f"Masked {mod_name} ({len(mask_indices[batch_idx])} patches)")
        fig.colorbar(im1, ax=axes[1])
        
        # Reconstructed image
        im2 = axes[2].imshow(reconstructed_image[batch_idx, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[2].set_title(f"Reconstructed {mod_name}")
        fig.colorbar(im2, ax=axes[2])
        
        # Calculate PSNR and MSE for the full image
        mse = torch.nn.functional.mse_loss(original_image, reconstructed_image).item()
        max_signal = torch.max(original_image).item()
        if mse > 1e-10:
            psnr = 20 * np.log10(max_signal / np.sqrt(mse))
        else:
            psnr = 100
        
        plt.suptitle(f"{mod_name} Full Image Reconstruction - Batch {batch_idx}\nMSE: {mse:.6f}, PSNR: {psnr:.2f} dB", y=1.05, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"full_reconstruction_{mod_name}_batch{batch_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also visualize reconstruction error
        error_map = torch.abs(original_image - reconstructed_image)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im_error = ax.imshow(error_map[batch_idx, 0, center_slice].cpu().numpy(), cmap='hot')
        ax.set_title(f"Reconstruction Error - {mod_name}\nMax Error: {error_map.max().item():.6f}")
        fig.colorbar(im_error, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"error_map_{mod_name}_batch{batch_idx}.png"), dpi=300)
        plt.close()
        
        # Create an animated GIF of all slices (optional)
        all_slices = []
        for d in range(D):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(original_image[batch_idx, 0, d].cpu().numpy(), cmap='gray')
            axes[0].set_title(f"Original {mod_name} - Slice {d}")
            
            axes[1].imshow(masked_images[batch_idx, 0, d].cpu().numpy(), cmap='gray')
            axes[1].set_title(f"Masked {mod_name} - Slice {d}")
            
            axes[2].imshow(reconstructed_image[batch_idx, 0, d].cpu().numpy(), cmap='gray')
            axes[2].set_title(f"Reconstructed {mod_name} - Slice {d}")
            
            plt.tight_layout()
            plt.suptitle(f"{mod_name} Full Image Reconstruction - Slice {d} of {D}", y=1.02)
            
            # Save to buffer
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            all_slices.append(image)
            plt.close()
        
        # Save as GIF (requires imageio)
        try:
            import imageio
            imageio.mimsave(os.path.join(save_dir, f"reconstruction_animation_{mod_name}_batch{batch_idx}.gif"), 
                           all_slices, fps=5)
            print(f"Saved animation for {mod_name}")
        except ImportError:
            print("imageio not installed. Skipping animation creation.")
        
        print(f"Modality: {mod_name}")
        print(f"Full Image MSE: {mse:.6f}")
        print(f"Full Image PSNR: {psnr:.2f} dB")
        print(f"Number of masked patches: {len(mask_indices[batch_idx])}")
        print(f"Max reconstruction error: {error_map.max().item():.6f}")
        print("-" * 50)

def reconstruct_full_image(masked_images, reconstructed_patches, mask_indices, patch_positions, patch_size, image_size):
    """
    Reconstruct the full image from masked images and reconstructed patches
    
    Args:
        masked_images: Images with masked patches [B, C, D, H, W]
        reconstructed_patches: Reconstructed patches [B, num_patches, patch_volume]
        mask_indices: List of indices that were masked for each batch
        patch_positions: The 3D position of each patch
        patch_size: Size of each patch (D, H, W)
        image_size: Size of the full image (D, H, W)
    
    Returns:
        reconstructed_images: Full reconstructed images [B, C, D, H, W]
    """
    B, C, D, H, W = masked_images.shape
    pd, ph, pw = patch_size
    
    # Start with the masked images
    reconstructed_images = masked_images.clone()
    
    # Place reconstructed patches back into the correct positions
    for b in range(B):
        if b < len(mask_indices):
            masked_idx = mask_indices[b]
            for patch_idx in masked_idx:
                # Check if index is valid
                if patch_idx < len(patch_positions) and patch_idx < reconstructed_patches.shape[1]:
                    # Get the patch position
                    d_idx, h_idx, w_idx = patch_positions[patch_idx]
                    
                    # Calculate starting position in the original image
                    d_start = d_idx * pd
                    h_start = h_idx * ph
                    w_start = w_idx * pw
                    
                    # Check boundaries to avoid index errors
                    if d_start + pd <= D and h_start + ph <= H and w_start + pw <= W:
                        # Reshape reconstructed patch from flattened to 3D
                        # For single channel, patch should be reshaped to (pd, ph, pw)
                        patch = reconstructed_patches[b, patch_idx].reshape(pd, ph, pw)
                        
                        # Place patch back in the image
                        reconstructed_images[b, 0, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] = patch
    
    return reconstructed_images


def visualize_cross_modal_full_reconstruction(original_images_dict, cross_recon_dict, mask_indices_list, 
                                            patch_positions_list, modality_names=["T1c", "T1n", "T2f", "T2w"],
                                            batch_idx=0, save_dir="reconstruction_visualizations", 
                                            image_size=(128, 128, 128)):
    """
    Visualize full image cross-modal reconstruction by properly reconstructing entire images
    
    Args:
        original_images_dict: Dictionary with modality indices as keys and original images as values
        cross_recon_dict: Dictionary with cross-modal reconstruction patches
        mask_indices_list: List of mask indices for each modality
        patch_positions_list: List of patch positions for each modality
        modality_names: Names of the modalities
        batch_idx: Which batch to visualize
        save_dir: Directory to save visualizations
        image_size: Size of the images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get actual tensor dimensions from one of the original images
    C = 1  # Single channel
    patch_size = (4, 16, 16)
    pd, ph, pw = patch_size
    patch_volume = C * pd * ph * pw  # 1 * 4 * 16 * 16 = 1024
    
    for cross_key, cross_patches in cross_recon_dict.items():
        source_idx, target_idx = map(int, cross_key.split('_to_'))
        source_name = modality_names[source_idx]
        target_name = modality_names[target_idx]
        
        # Get the mask indices from source modality
        mask_indices = mask_indices_list[source_idx]
        # Use the first modality's patch positions (should be same across all modalities)
        patch_positions = patch_positions_list[0]
        
        # Get the original target image
        original_target_image = original_images_dict[target_idx]
        
        # Create a masked version of the target image based on source modality's mask
        masked_target_image = original_target_image.clone()
        actual_D, actual_H, actual_W = original_target_image.shape[2:]
        
        if batch_idx < len(mask_indices):
            for patch_idx in mask_indices[batch_idx]:
                if patch_idx < len(patch_positions):
                    d_idx, h_idx, w_idx = patch_positions[patch_idx]
                    d_start = d_idx * pd
                    h_start = h_idx * ph
                    w_start = w_idx * pw
                    
                    # Check boundaries to avoid index errors
                    if d_start + pd <= actual_D and h_start + ph <= actual_H and w_start + pw <= actual_W:
                        masked_target_image[0, :, d_start:d_start+pd, 
                                           h_start:h_start+ph, 
                                           w_start:w_start+pw] = 0.0
        
        # Reconstruct full image by placing cross-modal reconstructed patches
        reconstructed_target_image = masked_target_image.clone()
        
        # Get the correct mask indices (from source modality)
        source_mask_indices = mask_indices[batch_idx] if batch_idx < len(mask_indices) else []
        
        # Place reconstructed patches in the correct positions
        for idx_in_mask, patch_idx in enumerate(source_mask_indices):
            if patch_idx < len(patch_positions) and idx_in_mask < cross_patches.shape[1]:
                d_idx, h_idx, w_idx = patch_positions[patch_idx]
                d_start = d_idx * pd
                h_start = h_idx * ph
                w_start = w_idx * pw
                
                # Check boundaries to avoid index errors
                if d_start + pd <= actual_D and h_start + ph <= actual_H and w_start + pw <= actual_W:
                    # Reshape from flattened to 3D
                    patch = cross_patches[batch_idx, idx_in_mask].reshape(C, pd, ph, pw)
                    reconstructed_target_image[0, :, d_start:d_start+pd, 
                                             h_start:h_start+ph, 
                                             w_start:w_start+pw] = patch
        
        # Visualize
        # Get actual tensor dimensions instead of using the expected image size
        actual_D = original_target_image.shape[2]
        center_slice = actual_D // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original target image
        im0 = axes[0].imshow(original_target_image[0, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[0].set_title(f"Original {target_name}")
        fig.colorbar(im0, ax=axes[0])
        
        # Masked target image
        im1 = axes[1].imshow(masked_target_image[0, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[1].set_title(f"Masked {target_name} (using {source_name} mask)")
        fig.colorbar(im1, ax=axes[1])
        
        # Cross-reconstructed image
        im2 = axes[2].imshow(reconstructed_target_image[0, 0, center_slice].cpu().numpy(), cmap='gray')
        axes[2].set_title(f"{source_name} → {target_name} Reconstruction")
        fig.colorbar(im2, ax=axes[2])
        
        # Calculate metrics only for the masked region
        mse_list = []
        for patch_idx in source_mask_indices:
            if patch_idx < len(patch_positions):
                d_idx, h_idx, w_idx = patch_positions[patch_idx]
                d_start, h_start, w_start = d_idx * pd, h_idx * ph, w_idx * pw
                
                if d_start + pd <= actual_D and h_start + ph <= actual_H and w_start + pw <= actual_W:
                    # Compare original and reconstructed patches
                    orig_patch = original_target_image[0, 0, d_start:d_start+pd, 
                                                     h_start:h_start+ph, w_start:w_start+pw]
                    recon_patch = reconstructed_target_image[0, 0, d_start:d_start+pd, 
                                                           h_start:h_start+ph, w_start:w_start+pw]
                    mse_list.append(F.mse_loss(orig_patch, recon_patch).item())
        
        # Calculate overall PSNR
        if mse_list:
            mse = sum(mse_list) / len(mse_list)
            max_signal = torch.max(original_target_image).item()
            if mse > 1e-10:
                psnr = 20 * np.log10(max_signal / np.sqrt(mse))
            else:
                psnr = 100
        else:
            mse = 0
            psnr = 100
        
        plt.suptitle(f"Cross-Modal Full Image Reconstruction: {source_name} → {target_name}\nMSE: {mse:.6f}, PSNR: {psnr:.2f} dB", y=1.05, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cross_modal_full_{source_name}_to_{target_name}_batch{batch_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create error map visualization
        error_map = torch.abs(original_target_image - reconstructed_target_image)
        actual_D = error_map.shape[2]
        center_slice = actual_D // 2
        error_center_slice = error_map[0, 0, center_slice].cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im_error = ax.imshow(error_center_slice, cmap='hot', vmax=np.percentile(error_center_slice, 99))
        ax.set_title(f"Cross-Modal Reconstruction Error - {source_name} → {target_name}\nMax Error: {error_map.max().item():.6f}, Mean Error: {error_map.mean().item():.6f}")
        fig.colorbar(im_error, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cross_modal_error_map_{source_name}_to_{target_name}_batch{batch_idx}.png"), dpi=300)
        plt.close()
        
        print(f"Cross-Modal: {source_name} → {target_name}")
        print(f"Masked Region MSE: {mse:.6f}")
        print(f"Masked Region PSNR: {psnr:.2f} dB")
        print(f"Number of masked patches: {len(source_mask_indices)}")
        print(f"Max reconstruction error: {error_map.max().item():.6f}")
        print(f"Mean reconstruction error: {error_map.mean().item():.6f}")
        print("-" * 50)


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_aux_v6_seed0.json"
    model_path = "/local2/jrgan/NephrologyKG/M3D/vision_aux_output_recon_model_name_pretrained_ViT.bin_freeze_vision_False_epochs_30_keep_only_bbox_False_bbox_loss_bce_use_cls_True_mask0.31121_selfonly_1011_m0.3_mim/best_model.pt"
    # model_path = "/local2/jrgan/NephrologyKG/M3D/vision_aux_output_recon_model_name_pretrained_ViT.bin_freeze_vision_False_epochs_30_keep_only_bbox_False_bbox_loss_bce_use_cls_True_mask0.31121_01.2_21.5_1011_m0.3_mim/best_model.pt"
    base_model_path = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
    output_dir = "visualization_results_full"
    os.makedirs(output_dir, exist_ok=True)
    
    class VisionConfig:
        vision_select_layer: int = -1
        vision_select_feature: str = "cls_patch"
        image_channel: int = 1
        image_size: list = [128, 128, 128]
        patch_size: list = [4, 16, 16]
        mask_ratio: float = 0.3
        
    print("Loading base model...")
    base_model = LamedPhi3ForCausalLM.from_pretrained(base_model_path)
    
    # Get vision tower
    vision_tower = base_model.get_model().get_vision_tower()
    vision_tower.select_feature = 'cls_patch'  # Set to use CLS token
    vision_tower.mask_ratio = 0.3  # Set masking ratio

    
    # Define cross-modality matrix (same as in training)
    cross_modality_matrix = torch.zeros((4, 4))
    for i in range(4):
        cross_modality_matrix[i][i] = 0.0
    # cross_modality_matrix[2, 0] = 1.5  # T2f -> T1c
    # cross_modality_matrix[2, 1] = 1.5  # T2f -> T1n 
    # cross_modality_matrix[2, 3] = 1.5  # T2f -> T2w
    # cross_modality_matrix[0, 1] = 1.2  # T2f -> T1c
    # cross_modality_matrix[0, 2] = 1.2  # T2f -> T1n 
    # cross_modality_matrix[0, 3] = 1.2  # T2f -> T2w
    
    # Initialize model
    print("Initializing model...")
    model = VisionAuxClassifier(
        vision_tower=vision_tower,
        num_modalities=4,
        area_levels=10,
        extent_levels=6,
        solidity_levels=4,
        num_quadrants=27,
        use_cls=True,
        modality_weights=[1.0, 1.0, 2.0, 1.0],
        cross_modality_matrix=cross_modality_matrix
    ).to(device)
    
    # Load trained weights
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset
    print(f"Loading test data from {test_file}")
    test_dataset = AuxVisionDataset(test_file, mode="test", num_quadrants=27)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Select a random example from the test set
    example_idx = random.randint(0, len(test_dataset) - 1)
    print(f"Visualizing example {example_idx} from test set")
    
    # Get the example
    example = test_dataset[example_idx]
    
    # Create a batch of one example
    mod1 = example["t1c"].unsqueeze(0).to(device)  # T1-contrast
    mod2 = example["t1n"].unsqueeze(0).to(device)  # T1-native
    mod3 = example["t2f"].unsqueeze(0).to(device)  # T2-FLAIR
    mod4 = example["t2w"].unsqueeze(0).to(device)  # T2-weighted
    
    print("Performing reconstruction...")
    # Run MIM
    with torch.no_grad():
        mim_results = model.vision_tower.forward_mim_multimodal(
            mod1, mod2, mod3, mod4, 
            apply_mask=True
        )
    
    # Prepare original images dictionary
    original_images_dict = {
        0: mod1,
        1: mod2,
        2: mod3,
        3: mod4
    }

        # Visualize individual patches
    print("Visualizing individual patches...")
    visualize_multiple_patches(
        mim_results['modality_results'],
        mim_results['cross_reconstructed'],
        modality_names=["T1c", "T1n", "T2f", "T2w"],
        batch_idx=0,
        num_patches_to_show=5,
        patch_size=(4, 16, 16),
        save_dir=output_dir
    )
    
    
    # Visualize full image reconstruction
    print("Visualizing full image reconstruction...")
    visualize_full_reconstruction(
        mim_results['modality_results'],
        modality_names=["T1c", "T1n", "T2f", "T2w"],
        batch_idx=0,
        save_dir=output_dir,
        device=device
    )
    
    # Visualize cross-modal full image reconstruction
    if 'cross_reconstructed' in mim_results:
        print("Visualizing cross-modal full image reconstruction...")
        mask_indices_list = [x['mask_indices'] for x in mim_results['modality_results']]
        patch_positions_list = [x['patch_positions'] for x in mim_results['modality_results']]
        
        visualize_cross_modal_full_reconstruction(
            original_images_dict,
            mim_results['cross_reconstructed'],
            mask_indices_list,
            patch_positions_list,
            modality_names=["T1c", "T1n", "T2f", "T2w"],
            batch_idx=0,
            save_dir=output_dir,
            image_size=(128, 128, 128)
        )
    
    print(f"Full reconstruction visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
