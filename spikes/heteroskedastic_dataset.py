#%%
import blackhc.project.script
from io import BytesIO

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#%%

def load_image(fsspec_path):
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"
    }
    # Open a connection to the URL and load the image
    with fsspec.open(fsspec_path, mode='rb', headers=headers) as f:
        img_data = f.read()
        img = Image.open(BytesIO(img_data))
        img.load()  # Make sure PIL has read the data
    return img

# Example usage 
url = "dss_sprint/datasets/heteroskedastic/blackhc_satellite_view_of_oxford_30cm_5x5km_ee9500e6-13cc-49c6-84e7-278466d7fa9f.png"
image = load_image(url)

#%%
# Print the resolution of the image
print(image.size)

# %%
# Downscale to 256x256 and plot
downscaled_image = image.resize((256, 256))
downscaled_image.show()

# %%
# Convert to grayscale
grayscale_image = image.convert('L')
grayscale_image.show()

# %%
# Convert to numpy array
image_array = np.array(grayscale_image) / 255.

# %% Reshape into 256x256 tiles of 16x16
image_array = image_array.reshape((256, 8, 256, 8)).transpose((0,2,1,3)).reshape((256, 256, -1))

# Compute the mean and std of each tile
tile_means = image_array.mean(axis=(-1))
tile_stds = image_array.std(axis=(-1))

# Plot the means and stds
plt.imshow(tile_means, cmap='gray')
plt.show()
plt.imshow(tile_stds, cmap='gray')
plt.show()

# %%
from PIL import Image
import numpy as np

def reshape_image_into_tiles(image: Image.Image, tile_size: int) -> np.ndarray:
    """
    Reshape and permute a PIL image into tiles of a certain size.

    Parameters:
    - image: PIL.Image.Image - The input image. [H, W, C]
    - tile_size: int - The size of each tile.

    Returns:
    - np.ndarray - The reshaped and permuted image as a numpy array. [C, H, W, T, T]
    """
    # Convert the image to a numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0
    
    # Get the dimensions of the image
    height, width = image_array.shape[:2]
    
    # Ensure the dimensions are divisible by the tile size
    assert height % tile_size == 0, "Height must be divisible by tile size"
    assert width % tile_size == 0, "Width must be divisible by tile size"
    
    # Reshape and permute the image array
    reshaped_array = image_array.reshape(
        (height // tile_size, tile_size, width // tile_size, tile_size, -1)
    ).transpose((4, 0, 2, 1, 3)).reshape(
        (-1, height // tile_size, width // tile_size, tile_size, tile_size)
    )
    
    return reshaped_array

def reshape_tiles_into_histogram(image_tiles: np.ndarray) -> np.ndarray:
    """
    Merge the last two dimensions.
    """
    return image_tiles.reshape(image_tiles.shape[:-2] + (-1,))

# Example usage
tile_size = 16
reshaped_image = reshape_image_into_tiles(image, tile_size)
print(reshaped_image.shape)

# %%
sub_histograms = reshape_tiles_into_histogram(reshaped_image)
print(sub_histograms.shape)

# %%
mean = sub_histograms.mean(axis=-1)
std = sub_histograms.std(axis=-1)

# %%
plt.imshow(mean.transpose(1, 2, 0))
plt.show()
plt.imshow(std.transpose(1, 2, 0))
plt.show()

#%% Sample 1 pixel from each tile randomly
random_indices = np.random.randint(0, tile_size**2, sub_histograms.shape[1:3])
meshgrid = np.meshgrid(*[np.arange(i) for i in sub_histograms.shape[0:3]], indexing="ij")
sampled_pixels = sub_histograms[meshgrid[0], meshgrid[1], meshgrid[2], random_indices[None,...]]

# Plot the sampled pixels
plt.imshow(sampled_pixels.transpose(1, 2, 0))
plt.show()

# %%



# %%
# Plot histograms for each tile in sub_histograms
import matplotlib.pyplot as plt

for i in range(sub_histograms.shape[1]):
        plt.hist(sub_histograms[0, i, 0].flatten(), bins=256, alpha=0.5)
plt.show()

# %%

