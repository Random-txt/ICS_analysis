import numpy as np
import scipy.signal as signal
import tifffile as tiff
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from Fit_ICS import ICS_fit
from typing import Tuple, List
import os
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GlobalImageProcessor:
    def __init__(self, file_path: str, gaussian_window: int, gaussian_sigma: float, conv_factor: float, advanced_fitting: bool):
        self.file_path = file_path
        self.gaussian_window = gaussian_window
        self.gaussian_sigma = gaussian_sigma
        self.conv_factor = conv_factor
        self.filter_kernel = self.create_gauss_filter()
        self.raw_frames = self.extract_frames_from_tif() / conv_factor
        self.advanced_fitting = advanced_fitting

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

    def process_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, np.ndarray]:
        """Apply Gaussian filter, normalization, and autocorrelation to each frame."""
        gaussian_filter_frames = []
        filtered_frames = []
        normalized_frames = []
        autocorrelation_frames = []
        radial_autocorrelation_frames = []
        results_frames = []

        for frame in self.raw_frames:
            gauss_filter = self.gaussian_filter(frame)
            filtered_frame = self.safe_divide(frame, gauss_filter)
            normalized_frame = filtered_frame - np.nanmean(filtered_frame)


            gaussian_filter_frames.append(gauss_filter)
            filtered_frames.append(filtered_frame)
            normalized_frames.append(normalized_frame)

            autocorrelation_frame = self.AutoCorrelation(normalized_frame, filtered_frame)
            autocorrelation_frames.append(autocorrelation_frame)

            radii, radial_profile = self.calculate_radial_average(autocorrelation_frame)
            radial_autocorrelation_frames.append(radial_profile)

            if self.advanced_fitting:
                logging.info("Performing advanced fitting...")
                
                results = ICS_fit(autocorrelation_frame)
                results_frames.append(results)

                radial_fit = ICS_fit(radial_profile)
                results_frames.append(radial_fit)


        mean_autocorrelation_frame = np.mean(autocorrelation_frames, axis=0)
        mean_result = ICS_fit(mean_autocorrelation_frame)

        mean_radial_frame = np.mean(radial_autocorrelation_frames, axis=0)
        mean_radial_fit = ICS_fit(mean_radial_frame)

        results_frames.append(mean_result)
        results_frames.append(mean_radial_fit)


        return (
            np.array(gaussian_filter_frames),
            np.array(filtered_frames),
            np.array(normalized_frames),
            np.array(autocorrelation_frames),
            results_frames,
            radial_autocorrelation_frames,
            radii
        )

    def calculate_radial_average(self, normalized_autocorrelation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the radial average of a 2D autocorrelation."""
        center = np.array(normalized_autocorrelation.shape) // 2
        y, x = np.ogrid[:normalized_autocorrelation.shape[0], :normalized_autocorrelation.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
        radial_profile = np.bincount(r.ravel(), weights=normalized_autocorrelation.ravel()) / np.bincount(r.ravel())
        return np.arange(len(radial_profile)), radial_profile

    def AutoCorrelation(self, normalised_frame: np.ndarray, filtered_frame: np.ndarray) -> np.ndarray:
        autocorrelation = self.fft_autocorrelation_basic(normalised_frame)
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

    def plot_radial_profile_and_fit(self, radii: np.ndarray, radial_profile: np.ndarray, radial_fit: np.ndarray):
        """Plot radial profile and corresponding fit."""
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=radii, y=radial_profile, label='Radial Profile')
        sns.lineplot(x=radii, y=radial_fit, label='Fitted Curve', linestyle='--')
        plt.xlabel('Radius')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.title('Radial Profile and Fit')
        plt.legend()
        plt.show()

    def radial(self, results_frames: List, radii: np.ndarray) -> Tuple[List, List, List, List, List]:
        """Extract and process fit parameters and results."""
        
        # Initialize the lists outside the if block to be used in both cases
        AC_2D_models = []
        numberofmolecules = []
        waist_list = []
        debug_info = []
        radial_fits = []

        if self.advanced_fitting:
            # Process the arrays at even indices (0, 2, 4, ..., 18) for autocorrelation fit
            for i in range(0, 20, 2):  # Only even indices till 19
                AC_2D_model = results_frames[i][0]  # First element of the even-indexed array
                n_parameters = results_frames[i][1][0]
                w_parameters = results_frames[i][1][1]
                debug_information = results_frames[i][2]  # Third element

                AC_2D_models.append(AC_2D_model)
                numberofmolecules.append(n_parameters)
                waist_list.append(w_parameters)
                debug_info.append(debug_information)

            # Process the arrays at odd indices for radial fits (1, 3, 5, ...)
            radial_fits = [results_frames[i] for i in range(1, 20, 2)]
            
        else:
            # Non-advanced fitting: Only process the first frame's results
            AC_2D_model = results_frames[0][0]
            n_parameters = results_frames[0][1][0]
            w_parameters = results_frames[0][1][1]
            debug_information = results_frames[0][2]
            radial_fit = results_frames[1][0]

            # Append the results to the initialized lists
            AC_2D_models.append(AC_2D_model)
            numberofmolecules.append(n_parameters)  # Fixed: from npappend to append
            waist_list.append(w_parameters)
            debug_info.append(debug_information)
            radial_fits.append(radial_fit)
        
        return AC_2D_models, numberofmolecules, waist_list, debug_info, radial_fits


    def compute_global_stats(self, numberofmolecules: List[float], waist_list: List[float], results_frames: List):
        """Calculate global statistics for molecules and waist."""

        if self.advanced_fitting:
            # Calculate full global statistics when advanced fitting is enabled
            cr_glob = np.nanmean(self.raw_frames)
            cr_glob_std = np.nanstd(self.raw_frames)
            nmol_glob = results_frames[-2][1][0]
            waist_glob = results_frames[-2][1][1]
            crm_glob = cr_glob / nmol_glob
            
            
            cr_moy_glob = np.nanmean(self.raw_frames, axis=(1, 2))
            nmol_moy_glob = np.nanmean(numberofmolecules)
            waist_moy = np.nanmean(waist_list)
            crm_moy_glob = cr_moy_glob / nmol_moy_glob

            cr_std_glob = np.nanstd(self.raw_frames, axis=(1, 2))
            nmol_std_glob = np.nanstd(numberofmolecules)
            waist_std = np.nanstd(waist_list)
            crm_std_glob = np.nan  # Set NaN in case nmol_std_glob is NaN or zero

            # Compute crm_std_glob if nmol_std_glob is not zero
            if nmol_std_glob > 0:
                crm_std_glob = cr_glob / nmol_std_glob
            else:
                crm_std_glob = np.nan

        else:
            # Set global statistics to NaN if advanced fitting is disabled
            cr_glob = np.nanmean(self.raw_frames)
            cr_glob_std = np.nan
            nmol_glob = results_frames[-2][1][0]
            waist_glob = results_frames[-2][1][1]
            crm_glob = cr_glob / nmol_glob
            cr_glob_std = np.nan
            cr_moy_glob = np.nan
            nmol_moy_glob = np.nan
            waist_moy = np.nan
            crm_moy_glob = np.nan
            cr_std_glob = np.nan
            nmol_std_glob = np.nan
            waist_std = np.nan
            crm_std_glob = np.nan
            
            

        return {
            'CR_glob': cr_glob,
            'CR_glob_std': cr_glob_std,
            'CR_moy_glob': cr_moy_glob,
            'CR_std_glob': cr_std_glob,
            'N_glob': nmol_glob,
            'N_moy_glob': nmol_moy_glob,
            'N_std_glob': nmol_std_glob,
            'Waist_glob': waist_glob,
            'Waist_moy': waist_moy,
            'Waist_std': waist_std,
            'CRM_glob': crm_glob,
            'CRM_std_glob': crm_std_glob,
            'CRM_moy_glob': crm_moy_glob
        }

# def main():
#     # Set parameters
                
#     gaussian_window = 121
#     gaussian_sigma = 40
#     dtime = 1E-5
#     conv_factor = dtime / 1E3
    
#     file_directory = "images"
#     for file in os.listdir(file_directory):
#         if file.endswith(".tif"):
#             file_path = os.path.join(file_directory, file)
            
#             # Create an instance of the GlobalImageProcessor class
#             processor = GlobalImageProcessor(file_path, gaussian_window, gaussian_sigma, conv_factor)
#             # Process frames and get the results
#             gaussian_frames, filtered_frames, normalized_frames, autocorrelation_frames, results_frames, radial_autocorrelation_frames, radii = processor.process_frames(advanced_fitting=False)
#             # Perform fitting and get models
#             AC_2D_models, numberofmolecules, waist_list, debug_info, radial_fits = processor.radial(results_frames, radii)
#             # Compute global statistics
#             global_stats = processor.compute_global_stats(numberofmolecules, waist_list, results_frames)
#             logging.info("Global processing finished.")

# #

# if __name__ == "__main__":
#     main()
