import torch
import numpy as np
import torchvision.transforms.functional as TF

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib is not installed. Visualization functions will not work. \n"
          "Please install it using 'pip install matplotlib'")
    MATPLOTLIB_AVAILABLE = False

try:
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image is not installed. Visualization functions may not work as expected. \n"
          "Please install it using 'pip install scikit-image'")
    SKIMAGE_AVAILABLE = False

# Module imports
from suit import ParameterPredictor # Changed
from utils import generate_superpixels # Changed
from suit import suit_tiny_224_adaptive, SuitAdaptive # Changed
import utils # For accessing utils.SKIMAGE_AVAILABLE

# Define typical ImageNet mean and std if needed for unnormalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def simple_unnormalize_image(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """
    Reverses the ImageNet normalization for a single image tensor.
    Assumes tensor is [C, H, W].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (C, H, W).")
    
    unnormalized_tensor = tensor.clone()
    for c in range(unnormalized_tensor.shape[0]):
        unnormalized_tensor[c] = unnormalized_tensor[c] * std[c] + mean[c]
    return torch.clamp(unnormalized_tensor, 0, 1)


def plot_superpixel_overlay(image_tensor: torch.Tensor, 
                            superpixel_map: torch.Tensor, 
                            K_predicted: float = None, 
                            m_predicted: float = None, 
                            ax=None, 
                            title_prefix="Superpixels"):
    """
    Plots an image with superpixel boundaries overlaid.

    Args:
        image_tensor (torch.Tensor): Single image tensor [C, H, W].
        superpixel_map (torch.Tensor): Superpixel map [H, W].
        K_predicted (float, optional): Predicted K value.
        m_predicted (float, optional): Predicted m value.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses current axes.
        title_prefix (str): Prefix for the plot title.
    """
    if not MATPLOTLIB_AVAILABLE or not SKIMAGE_AVAILABLE:
        print("Matplotlib or scikit-image not available. Skipping plot.")
        return

    # Unnormalize and convert image for display
    img_display = simple_unnormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    # Ensure image is float in [0, 1] for mark_boundaries if it's not already
    img_display_float = img_as_float(img_display) 

    sp_map_np = superpixel_map.cpu().numpy()

    if ax is None:
        ax = plt.gca()

    ax.imshow(mark_boundaries(img_display_float, sp_map_np, color=(1,0,0))) # Red boundaries
    title = title_prefix
    if K_predicted is not None and m_predicted is not None:
        title += f" (K={K_predicted:.1f}, m={m_predicted:.1f})"
    elif K_predicted is not None:
         title += f" (K={K_predicted:.1f})"
    ax.set_title(title)
    ax.axis('off')


def compare_fixed_vs_adaptive_segmentation(image_tensor: torch.Tensor, 
                                           adaptive_model: SuitAdaptive, 
                                           fixed_K: int = 200, 
                                           fixed_m: float = 10.0,
                                           output_path: str = None):
    """
    Compares superpixel segmentation using fixed parameters vs. adaptive parameters
    from the SuitAdaptive model.

    Args:
        image_tensor (torch.Tensor): Single image tensor [C, H, W].
        adaptive_model (SuitAdaptive): Instance of the SuitAdaptive model.
        fixed_K (int): Fixed number of superpixels.
        fixed_m (float): Fixed compactness value.
        output_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    if not MATPLOTLIB_AVAILABLE or not SKIMAGE_AVAILABLE:
        print("Matplotlib or scikit-image not available. Skipping comparison plot.")
        return
    
    if not utils.SKIMAGE_AVAILABLE: # Check skimage specifically for generate_superpixels - CHANGED
        print("scikit-image needed by generate_superpixels is not available. Skipping comparison.")
        return

    adaptive_model.eval() # Ensure model is in eval mode

    # --- Adaptive Segmentation ---
    # ParameterPredictor expects a batch, so add batch dimension
    image_batch = image_tensor.unsqueeze(0) 
    
    with torch.no_grad(): # Important for inference
        params = adaptive_model.parameter_predictor(image_batch)
    
    K_pred = params["num_superpixels"] # Shape [B] or [B,1]
    m_pred = params["compactness"]   # Shape [B] or [B,1]

    # Clip and normalize parameters as in SuitAdaptive.forward
    K_adaptive_tensor = torch.clamp(K_pred, min=50, max=1000)
    m_adaptive_tensor = torch.clamp(m_pred, min=1, max=40)

    # Ensure K and m are [B] for generate_superpixels
    if K_adaptive_tensor.dim() > 1 and K_adaptive_tensor.shape[1] == 1:
        K_adaptive_tensor = K_adaptive_tensor.squeeze(1)
    if m_adaptive_tensor.dim() > 1 and m_adaptive_tensor.shape[1] == 1:
        m_adaptive_tensor = m_adaptive_tensor.squeeze(1)

    sp_map_adaptive = generate_superpixels(image_batch, K_adaptive_tensor, m_adaptive_tensor).squeeze(0)
    
    K_adaptive_val = K_adaptive_tensor.item() # Get scalar value for title
    m_adaptive_val = m_adaptive_tensor.item() # Get scalar value for title

    # --- Fixed Segmentation ---
    K_fixed_tensor = torch.tensor([fixed_K], device=image_tensor.device, dtype=torch.float) # Match type if necessary
    m_fixed_tensor = torch.tensor([fixed_m], device=image_tensor.device, dtype=torch.float)
    sp_map_fixed = generate_superpixels(image_batch, K_fixed_tensor, m_fixed_tensor).squeeze(0)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    plot_superpixel_overlay(image_tensor, sp_map_adaptive, 
                            K_predicted=K_adaptive_val, m_predicted=m_adaptive_val, 
                            ax=axes[0], title_prefix="Adaptive Seg.")
    
    plot_superpixel_overlay(image_tensor, sp_map_fixed, 
                            K_predicted=fixed_K, m_predicted=fixed_m, 
                            ax=axes[1], title_prefix="Fixed Seg.")
    
    fig.suptitle("Superpixel Segmentation Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    if output_path:
        plt.savefig(output_path)
        print(f"Comparison plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    if not MATPLOTLIB_AVAILABLE or not SKIMAGE_AVAILABLE:
        print("Matplotlib or scikit-image is not available. Skipping example usage.")
    elif not utils.SKIMAGE_AVAILABLE: # Check skimage for generate_superpixels - CHANGED
        print("scikit-image (for generate_superpixels) not available. Skipping example usage that needs it.")
    else:
        print("Running visualization examples...")
        # Create a dummy image tensor (e.g., random, or a simple gradient)
        # This tensor should ideally be normalized like ImageNet data if using simple_unnormalize_image
        dummy_image_c, dummy_image_h, dummy_image_w = 3, 224, 224
        
        # Create a tensor with some pattern to make superpixels more visually distinct than pure random
        dummy_image_np = np.random.rand(dummy_image_h, dummy_image_w, dummy_image_c).astype(np.float32)
        for i in range(dummy_image_h // 2): # Simple gradient
            dummy_image_np[i,:,0] = (i*2) / dummy_image_h
        for j in range(dummy_image_w // 2):
            dummy_image_np[:,j,1] = (j*2) / dummy_image_w
        
        dummy_image_tensor = torch.from_numpy(dummy_image_np).permute(2,0,1) # C, H, W

        # Normalize it (example, if it were in [0,1] initially)
        normalize = TF.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_dummy_image = normalize(dummy_image_tensor)


        # --- Example 1: plot_superpixel_overlay ---
        print("Example 1: Basic plot_superpixel_overlay")
        # Generate some dummy superpixels for this example
        example_K = torch.tensor([150.0])
        example_m = torch.tensor([15.0])
        
        # generate_superpixels expects a batch
        dummy_sp_map = generate_superpixels(normalized_dummy_image.unsqueeze(0), example_K, example_m).squeeze(0)
        
        plt.figure(figsize=(6,6)) # Control figure size for single plot
        plot_superpixel_overlay(normalized_dummy_image, dummy_sp_map, 
                                K_predicted=example_K.item(), m_predicted=example_m.item(),
                                title_prefix="Single Overlay Example")
        if MATPLOTLIB_AVAILABLE: # only try to save if matplotlib is there
            plt.savefig("visualize/example_single_overlay.png")
            print("Saved visualize/example_single_overlay.png")
            plt.close() # Close the figure
        else:
            plt.show() # Fallback if savefig fails or not desired without output_path logic


        # --- Example 2: compare_fixed_vs_adaptive_segmentation ---
        print("\nExample 2: compare_fixed_vs_adaptive_segmentation")
        # Instantiate SuitAdaptive model (using a small variant for example)
        # Ensure num_classes is passed if the model requires it for the head.
        # For visualization, we don't need pretrained weights.
        adaptive_model_instance = suit_tiny_224_adaptive(num_classes=10, pretrained=False)
        adaptive_model_instance.eval()

        compare_fixed_vs_adaptive_segmentation(normalized_dummy_image, 
                                               adaptive_model_instance, 
                                               fixed_K=100, 
                                               fixed_m=20,
                                               output_path="visualize/example_comparison.png")
        
        print("\nVisualization examples finished.")
        print("Note: If plots are not showing, ensure your environment supports GUI display (e.g., not a headless server without X11 forwarding).")
        print("Saved images will be in 'visualize/' directory (if it exists or can be created).")

        # Create the visualize directory if it doesn't exist
        import os
        os.makedirs("visualize", exist_ok=True)

```
