import torch
import numpy as np

try:
    from skimage.segmentation import slic
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    # We'll print a message within the function if skimage is needed but not available.

def generate_superpixels(image_batch: torch.Tensor, K_batch: torch.Tensor, m_batch: torch.Tensor) -> torch.Tensor:
    """
    Generates superpixel maps for a batch of images using SLIC.

    Args:
        image_batch (torch.Tensor): A batch of images, shape [B, C, H, W].
                                    Assumed to be in a range suitable for skimage.slic (e.g., float [0,1] or [0,255]).
        K_batch (torch.Tensor): A tensor containing the number of superpixels for each image, shape [B].
        m_batch (torch.Tensor): A tensor containing the compactness parameter for each image, shape [B].

    Returns:
        torch.Tensor: A batch of superpixel maps, shape [B, H, W].
    """
    if not SKIMAGE_AVAILABLE:
        print("Warning: scikit-image is not installed. Superpixel generation will be skipped. \n"
              "Please install it using 'pip install scikit-image'")
        # Return a tensor of zeros or raise an error, depending on desired behavior.
        # For now, let's return zeros of the expected output shape if image_batch is provided.
        if image_batch is not None and image_batch.dim() == 4:
            B, _, H, W = image_batch.shape
            return torch.zeros((B, H, W), dtype=torch.long, device=image_batch.device)
        else:
            # Cannot determine shape, raise error or return empty tensor
            raise ImportError("scikit-image is not available and input image_batch is invalid for shape inference.")

    batch_size = image_batch.shape[0]
    superpixel_maps = []

    for i in range(batch_size):
        current_image_tensor = image_batch[i] # Shape: [C, H, W]
        num_superpixels = int(K_batch[i].item())
        compactness = float(m_batch[i].item()) # slic expects float for compactness

        # Convert image to NumPy array, channel-last (H, W, C)
        # Skimage slic expects a 2D image for grayscale or a 3D image with channels_last for color.
        # It also prefers double precision floats in range [0,1] or unsigned ints.
        # We'll assume the input tensor is float and permute.
        # If the image is grayscale (C=1), we might need to squeeze the channel dim after permute.
        
        if current_image_tensor.shape[0] == 1: # Grayscale
            # Permute to (H, W, C) and then squeeze to (H, W) if C=1
            img_np = current_image_tensor.permute(1, 2, 0).cpu().numpy().squeeze(axis=-1)
        else: # Color
            img_np = current_image_tensor.permute(1, 2, 0).cpu().numpy()

        # Ensure the image is C-contiguous, which slic might prefer for performance or require.
        # Values for slic are often expected to be in [0,1] for float images.
        # If not, skimage might convert internally, but it's good practice.
        # For now, we assume the input image_batch is already appropriately normalized.
        img_np = np.ascontiguousarray(img_np)

        # Call skimage.segmentation.slic
        # slic_zero=True can be useful to guarantee that label 0 is not present if needed.
        # For multichannel images, channel_axis=-1 is the default and correct here.
        superpixel_map = slic(img_np, n_segments=num_superpixels, compactness=compactness,
                              slic_zero=True, start_label=1) 
        superpixel_maps.append(torch.from_numpy(superpixel_map))

    # Stack the list of tensors into a single tensor
    # Each superpixel_map is [H, W], so stacking gives [B, H, W]
    batched_superpixel_maps = torch.stack(superpixel_maps, dim=0)

    return batched_superpixel_maps.to(image_batch.device) # Move to original device

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    if SKIMAGE_AVAILABLE:
        B, C, H, W = 2, 3, 64, 64
        dummy_images = torch.rand(B, C, H, W)  # Example images (normalized to [0,1])
        dummy_K = torch.tensor([100, 150])     # Number of superpixels for each image
        dummy_m = torch.tensor([10.0, 20.0])   # Compactness for each image

        superpixels_batch = generate_superpixels(dummy_images, dummy_K, dummy_m)
        print("Superpixels batch shape:", superpixels_batch.shape) # Expected: [B, H, W]
        print("Data type:", superpixels_batch.dtype)
        print("Device:", superpixels_batch.device)

        # Test with grayscale
        B, C, H, W = 2, 1, 64, 64
        dummy_gray_images = torch.rand(B, C, H, W)
        superpixels_gray_batch = generate_superpixels(dummy_gray_images, dummy_K, dummy_m)
        print("Grayscale superpixels batch shape:", superpixels_gray_batch.shape)


        # Test case where K and m might not be int/float initially
        dummy_K_float = torch.tensor([120.0, 130.0]) 
        dummy_m_int = torch.tensor([15, 25])
        superpixels_batch_mixed_types = generate_superpixels(dummy_images, dummy_K_float, dummy_m_int)
        print("Mixed types K,m superpixels batch shape:", superpixels_batch_mixed_types.shape)

    else:
        print("Skipping example usage as scikit-image is not available.")
        # Test the fallback behavior
        B, C, H, W = 2, 3, 64, 64
        dummy_images = torch.rand(B, C, H, W)
        dummy_K = torch.tensor([100, 150])
        dummy_m = torch.tensor([10.0, 20.0])
        superpixels_batch = generate_superpixels(dummy_images, dummy_K, dummy_m)
        print("Fallback superpixels batch shape:", superpixels_batch.shape) # Expected: [B, H, W]
        print("Fallback data type:", superpixels_batch.dtype)


    # Test with non-Tensor K and m (should still work due to .item())
    # This part is mostly conceptual as the type hints specify Tensors.
    # However, .item() would make it work if K_batch and m_batch were lists/arrays of single-element tensors.
    # For this test, K and m are simple Python lists of numbers.
    # The current implementation expects K_batch and m_batch to be Tensors.
    # If we wanted to support lists of numbers directly for K and m,
    # we would need to adjust how K_batch[i] and m_batch[i] are handled (no .item()).
    # For now, sticking to tensor inputs for K_batch and m_batch.
    # K_list = [100, 150]
    # m_list = [10.0, 20.0]
    # This would require changing K_batch[i].item() to K_batch[i] etc.
    # superpixels_batch_list_km = generate_superpixels(dummy_images, torch.tensor(K_list), torch.tensor(m_list))
    # print("List K,m superpixels batch shape:", superpixels_batch_list_km.shape)

```
