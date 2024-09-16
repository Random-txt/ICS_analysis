import numpy as np
import scipy.signal as signal
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from Fit_ICS import ICS_fit
from typing import Tuple, List, Any

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LocalImageProcessor:
    def __init__(self, file_path: str, gaussian_window: int, gaussian_sigma: float, conv_factor: float, grid_size: int, image_size: int):
        self.file_path = file_path
        self.gaussian_window = gaussian_window
        self.gaussian_sigma = gaussian_sigma
        self.conv_factor = conv_factor
        self.filter_kernel = self.create_gauss_filter()
        self.raw_frames = self.extract_frames_from_tif() / conv_factor
        self.subimage_grid_size = grid_size  # Use passed `grid_size` parameter
        self.subimage_size = image_size  # Use passed `image_size` parameter

    def create_gauss_filter(self) -> np.ndarray:
        """Create a 2D Gaussian filter."""
        ax = np.arange(-(self.gaussian_window // 2), self.gaussian_window // 2 + 1)
        X, Y = np.meshgrid(ax, ax)
        gauss = np.exp(-0.5 * (X ** 2 + Y ** 2) / self.gaussian_sigma ** 2)
        return gauss / np.sum(gauss)

    def gaussian_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to an image."""
        pad_width = self.filter_kernel.shape[0] // 2
        padded_frame = np.pad(frame, pad_width, mode='symmetric')
        filtered_frame = signal.fftconvolve(padded_frame, self.filter_kernel, mode='valid')
        return filtered_frame

    def interpolate_center_pixel(self, matrix: np.ndarray) -> np.ndarray:
        """Interpolate the center pixel of a matrix using surrounding pixels."""
        height, width = matrix.shape
        center_row, center_col = height // 2 - 1, width // 2 - 1
        surrounding_pixels = [
            matrix[center_row - 1, center_col],
            matrix[center_row + 1, center_col],
            matrix[center_row, center_col - 1],
            matrix[center_row, center_col + 1]
        ]
        matrix[center_row, center_col] = np.mean(surrounding_pixels)
        return matrix

    def safe_divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Safely divide two arrays, handling division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.divide(a, b)
            c[~np.isfinite(c)] = 0
        return c

    def extract_frames_from_tif(self) -> np.ndarray:
        """Open a multi-frame .tif file and return all frames as a single NumPy array."""
        return tiff.imread(self.file_path)

    def divide_frame_into_subimages(self, frame: np.ndarray, grid_size: int) -> List[np.ndarray]:
        """Divides a 2D frame into subimages based on a given grid size."""
        frame_height, frame_width = frame.shape
        subimage_height = frame_height // grid_size
        subimage_width = frame_width // grid_size
        subimages = []
        for row in range(0, frame_height, subimage_height):
            for col in range(0, frame_width, subimage_width):
                subimage = frame[row:row + subimage_height, col:col + subimage_width]
                subimages.append(subimage)
        return subimages

    def process_subimage_frames(self, advanced_fitting: bool = False):
        """Process subimage frames."""
        autocorrelated_subimages_grid = np.zeros((self.subimage_grid_size, self.subimage_grid_size, self.subimage_size, self.subimage_size))  # For one frame
        all_frames_autocorrelation = np.zeros((len(self.raw_frames), self.subimage_grid_size, self.subimage_grid_size, self.subimage_size, self.subimage_size))  # For all frames
        filtered_frames, normalized_frames, raw_subimage_frames = [], [], []
        filtered_subimage_means, normalized_subimage_means, raw_subimage_means = [], [], []
        subimage_results = []

        for frame_index, frame in enumerate(self.raw_frames):
            gauss_filter = self.gaussian_filter(frame)
            filtered_frame = self.safe_divide(frame, gauss_filter)
            normalized_frame = filtered_frame - np.nanmean(filtered_frame)
            normalized_subimages = self.divide_frame_into_subimages(normalized_frame, self.subimage_grid_size)
            filtered_subimages = self.divide_frame_into_subimages(filtered_frame, self.subimage_grid_size)
            raw_subimages = self.divide_frame_into_subimages(frame, self.subimage_grid_size)
            filtered_frames.append(filtered_frame)
            normalized_frames.append(normalized_frame)
            raw_subimage_frames.append(raw_subimages)

            filtered_means = [np.mean(subimage) for subimage in filtered_subimages]
            normalized_means = [np.mean(subimage) for subimage in normalized_subimages]
            raw_means = [np.mean(subimage) for subimage in raw_subimages]

            filtered_subimage_means.append(filtered_means)
            normalized_subimage_means.append(normalized_means)
            raw_subimage_means.append(raw_means)

            autocorrelated_subimages = []
            for norm_subimage, filt_subimage in zip(normalized_subimages, filtered_subimages):
                autocorr = self.AutoCorrelation(norm_subimage, filt_subimage)
                autocorrelated_subimages.append(autocorr)

            autocorrelated_subimages = np.array(autocorrelated_subimages).reshape(
                (self.subimage_grid_size, self.subimage_grid_size, self.subimage_size, self.subimage_size))
            all_frames_autocorrelation[frame_index] = autocorrelated_subimages

        mean_autocorrelation_subimage = np.mean(all_frames_autocorrelation, axis=0)
        for i in range(self.subimage_grid_size):
            for j in range(self.subimage_grid_size):
                subimage = mean_autocorrelation_subimage[i, j]
                result = ICS_fit(subimage)
                subimage_results.append(result)



        return (np.array(raw_subimage_means), np.array(filtered_subimage_means),
                np.array(normalized_subimage_means), subimage_results)

    def AutoCorrelation(self, normalized_frame: np.ndarray, filtered_frame: np.ndarray) -> np.ndarray:
        autocorrelation = self.fft_autocorrelation_basic(normalized_frame)
        autocorrelation = 1 + autocorrelation / (np.nanmean(filtered_frame)) ** 2
        return self.interpolate_center_pixel(autocorrelation)

    def fft_autocorrelation_basic(self, image: np.ndarray) -> np.ndarray:
        """Compute 2D autocorrelation using FFT."""
        N1, N2 = image.shape
        temp = np.zeros((2 * N1 - 1, 2 * N2 - 1))
        o1 = np.zeros((2 * N1 - 1, 2 * N2 - 1))
        o1[:N1, :N2] = 1

        C1 = np.real(np.fft.ifft2(np.fft.fft2(o1) * np.fft.fft2(o1)))
        temp[:N1, :N2] = image
        F = np.fft.fft2(temp)

        autocorrelation = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
        autocorrelation /= C1

        return np.real(autocorrelation[N1 - N1 // 2:N1 + N1 // 2, N2 - N2 // 2:N2 + N2 // 2])

    def plot_combined_heatmap(self, mean_autocorrelation_subimage):
        """Combine all subimages into a single large grid and plot as a heatmap."""
        combined_image = np.block([[mean_autocorrelation_subimage[i, j] for j in range(self.subimage_grid_size)] for i in range(self.subimage_grid_size)])
        plt.figure(figsize=(10, 10))
        sns.heatmap(combined_image, cmap='viridis', cbar=True)
        plt.title('Mean Autocorrelation Heatmap (8x8 Grid of 64x64 Subimages)')
        plt.axis('off')
        plt.show()

    def run(self):
        raw_subimage, filtered_subimage, normalized_subimage, subimage_results = self.process_subimage_frames(
            advanced_fitting=False)
        
        debug_success_list = []
        debug_nit_list = []
        debug_nfev_list = []
        number_of_molecules_list = []
        # Iterate over each subimage result
        for result in subimage_results:
            debug_info = result[2]  # Debug information dictionary
            number_of_molecules = result[1][0]
            
            number_of_molecules_list.append(number_of_molecules)
            
            # Extract success status from debug info
            success_status = debug_info.get('success', None)
            nit_status = debug_info.get('nit', None)
            nfev_status = debug_info.get('nfev', None)
            debug_success_list.append(success_status)
            debug_nit_list.append(nit_status)
            debug_nfev_list.append(nfev_status)
            
        number_of_molecules_array = np.array(number_of_molecules_list).reshape(self.subimage_grid_size, self.subimage_grid_size)
        debug_success_array = np.array(debug_success_list).reshape(self.subimage_grid_size, self.subimage_grid_size)
        debug_nit_array = np.array(debug_nit_list).reshape(self.subimage_grid_size, self.subimage_grid_size)
        debug_nfev_array = np.array(debug_nfev_list).reshape(self.subimage_grid_size, self.subimage_grid_size)
        
        return raw_subimage, filtered_subimage, normalized_subimage, number_of_molecules_array, debug_nit_array, debug_success_array, debug_nfev_array


    
    @staticmethod
    def extract_key_values(data_list: List[dict], key: str) -> List[Any]:
        """Extract values from a list of dictionaries."""
        return [d.get(key) for d in data_list if key in d]
