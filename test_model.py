import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import GAN_maker

# Function to unnormalize the output back to [0, 255] for displaying
def unnormalize_image(image):
    image = (image + 1) * 127.5  # Rescale from [-1, 1] to [0, 255]
    return np.clip(image, 0, 255).astype(np.uint8)

# Load saved model
checkpoint = torch.load('generator_checkpoint.pth')
generator = GAN_maker.Generator().to('cuda')
generator.load_state_dict(checkpoint['generator_state_dict'])

# Set generator to evaluation mode
generator.eval()

# Load only the last 5000 grayscale (L) and AB channel images
# Data preparation

# Load and reshape grayscale (L) data (only last 5000 images)
l_arr = np.reshape(np.load("data/l/gray_scale.npy")[-5000:], (5000, 224, 224, 1))  # Grayscale images

# Load AB channel data (for color) (last 5000 images across all batches)
arr2 = np.load("data/ab/ab/ab1.npy")[-1667:]  # Last 1667 images from batch 1
arr3 = np.load("data/ab/ab/ab2.npy")[-1667:]  # Last 1667 images from batch 2
arr4 = np.load("data/ab/ab/ab3.npy")[-1666:]  # Last 1666 images from batch 3

# Concatenate AB channels from different batches along the 0th axis (batch dimension)
ab_arr = np.concatenate([arr2, arr3, arr4], axis=0)

# Normalize both l_arr (grayscale) and ab_arr (color channels) to float32 and range [-1, 1]
l_arr = (l_arr / 127.5).astype(np.float32) - 1  # Normalize grayscale
ab_arr = (ab_arr / 127.5).astype(np.float32) - 1  # Normalize AB channels

# Convert numpy arrays to PyTorch tensors and change shape
l_tensor = torch.from_numpy(l_arr).permute(0,3,1,2)  # Shape: (5000, 1, 224, 224)
ab_tensor = torch.from_numpy(ab_arr).permute(0,3,1,2)  # Shape: (5000, 2, 224, 224)

# Create TensorDataset (Input: L channel, Target: AB channels)
test_dataset = TensorDataset(l_tensor, ab_tensor)  # Use last 5000 samples for testing

# Create DataLoader objects for testing with a smaller batch size
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Get a batch of test images
for batch_idx, (test_l, test_ab) in enumerate(test_loader):
    test_l = test_l.to('cuda')

    # Generate colorized images
    with torch.no_grad():
        generated_ab = generator(test_l)

    # Convert to CPU for displaying and unnormalize
    test_l = test_l.cpu().numpy()
    generated_ab = generated_ab.cpu().numpy()

    # Pick the first image from the batch
    grayscale_image = test_l[0][0]  # [0] is batch, [0] is L channel
    colorized_ab = generated_ab[0]  # [0] is the AB channels

    # Unnormalize grayscale (L) and colorized (AB) images
    grayscale_image = unnormalize_image(grayscale_image)
    colorized_ab = unnormalize_image(colorized_ab)

    # Concatenate grayscale with AB channels to get LAB image
    colorized_image = np.concatenate([grayscale_image[np.newaxis, :, :], colorized_ab], axis=0)
    colorized_image = np.transpose(colorized_image, (1, 2, 0))  # Change to HWC format

    # Convert LAB to RGB using OpenCV
    import cv2
    colorized_image_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2RGB)

    # Show the grayscale and colorized images
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Grayscale Image")

    plt.subplot(1, 2, 2)
    plt.imshow(colorized_image_rgb)
    plt.title("Colorized Image")

    plt.show()

    # Break after showing one batch (can be removed to process all images)
    break
