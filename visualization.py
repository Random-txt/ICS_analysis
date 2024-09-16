import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def plot_visualizations(
        raw_tif, gaussian_frames, filtered_frames, autocorrelation_frames, AC_2D_models, 
        radii, radial_autocorrelation_frames, radial_fits, raw_subimage, 
        filtered_subimage, normalized_subimage, Numberofmolecules_grid, 
        Brightness_grid, nfev_grid, success_grid, nit_grid, wait_mean, image_output_directory, image_name):
    """
    Function to plot multiple visualizations based on the provided data.
    """
    # First subplot: 2 rows, 3 columns
    plt.figure(figsize=(18, 12))
    
    # 1. Mean along axis=0 of raw_tif
    plt.subplot(2, 3, 1)
    plt.title("Raw Image")
    plt.imshow(np.mean(raw_tif, axis=0), cmap='viridis')
    plt.colorbar()

    # 2. Mean along axis=0 of gaussian_frames
    plt.subplot(2, 3, 2)
    plt.title("Gaussian Filter")
    plt.imshow(np.mean(gaussian_frames, axis=0), cmap='viridis')
    plt.colorbar()
    

    # 3. Mean along axis=0 of filtered_frames
    plt.subplot(2, 3, 3)
    plt.title("Filtered Image")
    plt.imshow(np.mean(filtered_frames, axis=0), cmap='viridis')
    plt.colorbar()

    
    # 4. Plot of radii vs mean of radial_autocorrelation_frames and radial_fits
    
    plt.subplot(2, 3, 4)
    plt.title(f"1D Autocorrelation - waist:{wait_mean:.3f}")
    plt.plot(radii, np.nanmean(radial_autocorrelation_frames, axis=0), label='Radial Autocorrelation') 
    plt.plot(radii, radial_fits[0], label='Radial Fits', linestyle='--')
    plt.xscale('log')
    plt.xlabel("Radii")
    plt.ylabel("G")
    plt.legend()

    
    # 5. Mean along axis=0 of autocorrelation_frames
    n = (np.mean(raw_tif, axis=0)).shape[0]
    start = (n - 30) // 2
    end = start + 30
    
    plt.subplot(2, 3, 5)
    plt.title("2D Autocorrelation")
    plt.imshow(np.mean(autocorrelation_frames, axis=0)[start:end, start:end], cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.title("2D Autocorrelation Fit")
    plt.imshow(np.mean(AC_2D_models, axis=0)[start:end, start:end], cmap='viridis')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{image_output_directory}/{image_name}_Figure_1_AC.png")
    # plt.show()

    # Second subplot: 2 rows, 3 columns
    plt.figure(figsize=(18, 12))

    # 1. Mean value of every frame in raw_tif vs frame number
    plt.subplot(2, 3, 1)
    plt.title("Intensity per frame")
    plt.plot(np.mean(raw_tif, axis=(1, 2)))
    plt.ylim(0, 1.1 * np.max(np.nanmean(raw_tif)))
    plt.xlabel("Frame number")
    plt.ylabel("Mean value")
    

    # 2. Heatmap of raw_subimage
    plt.subplot(2, 3, 2)
    plt.title("Tiff subimage")
    ax = sns.heatmap(raw_subimage, cmap='viridis', cbar=True)
    ax.set_aspect('equal')
    

    # 3. Heatmap of filtered_subimage
    plt.subplot(2, 3, 3)
    plt.title("Filtered subimage")
    ax = sns.heatmap(filtered_subimage, cmap='viridis', cbar=True)
    ax.set_aspect('equal')
    

    # 4. Heatmap of normalized_subimage
    plt.subplot(2, 3, 4)
    plt.title("Normalized subimage")
    ax = sns.heatmap(normalized_subimage, cmap='viridis', cbar=True)
    ax.set_aspect('equal')
    
    # 5. Heatmap of Numberofmolecules_grid
    plt.subplot(2, 3, 5)
    plt.title("Number of Molecules")
    ax = sns.heatmap(Numberofmolecules_grid, cmap='viridis', cbar=True)
    ax.set_aspect('equal')
    

    # 6. Heatmap of Brightness_grid
    plt.subplot(2, 3, 6)
    plt.title("Brightness")
    ax = sns.heatmap(Brightness_grid, cmap='viridis', cbar=True)
    ax.set_aspect('equal')


    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{image_output_directory}/{image_name}_Figure_2_Subimages.png")

    # Third subplot: 1 row, 3 columns
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("nfev grid")
    ax = sns.heatmap(nfev_grid, cmap='viridis', cbar=True)
    ax.set_aspect('equal')

    plt.subplot(1, 3, 2)
    plt.title("success grid")
    ax = sns.heatmap(success_grid, cmap='viridis', cbar=True)
    ax.set_aspect('equal')

    plt.subplot(1, 3, 3)
    plt.title("nit grid")
    ax = sns.heatmap(nit_grid, cmap='viridis', cbar=True)
    ax.set_aspect('equal')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{image_output_directory}/{image_name}_Figure_3_Debug.png")


    plt.close('all')
