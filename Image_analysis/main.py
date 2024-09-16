import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QCheckBox, QListWidget, QTabWidget
)
from PyQt6.QtCore import Qt
import logging
import numpy as np
import time

from Global_Analysis import GlobalImageProcessor
from Local_analysis import LocalImageProcessor
from save_excel import excel_save
from visualization import plot_visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input fields for parameters
        self.gaussian_window_label = QLabel("Gaussian Window:")
        self.gaussian_window_input = QLineEdit("121")
        layout.addWidget(self.gaussian_window_label)
        layout.addWidget(self.gaussian_window_input)

        self.gaussian_sigma_label = QLabel("Gaussian Sigma:")
        self.gaussian_sigma_input = QLineEdit("40")
        layout.addWidget(self.gaussian_sigma_label)
        layout.addWidget(self.gaussian_sigma_input)

        self.dtime_label = QLabel("Dtime:")
        self.dtime_input = QLineEdit("1E-5")
        layout.addWidget(self.dtime_label)
        layout.addWidget(self.dtime_input)

        # Drop-down for grid size and image size combinations
        self.grid_size_label = QLabel("Grid Size - Image Size:")
        self.grid_image_combo = QComboBox()
        self.grid_image_combo.addItems([
            "16 - 32",
            "8 - 64",
            "4 - 128",
            "2 - 256"
        ])
        self.grid_image_combo.setCurrentText("8 - 64")  # Set default value to "8 - 64"
        layout.addWidget(self.grid_size_label)
        layout.addWidget(self.grid_image_combo)

        # Checkbox for advanced fitting
        self.advanced_fitting_checkbox = QCheckBox("Enable Advanced Fitting (takes longer time)")
        layout.addWidget(self.advanced_fitting_checkbox)

        # File selection button for .tif files
        self.select_files_btn = QPushButton("Select .tif Images")
        self.select_files_btn.clicked.connect(self.select_files)
        layout.addWidget(self.select_files_btn)

        self.file_paths_label = QLabel("No files selected")
        layout.addWidget(self.file_paths_label)

        # Output directory selection
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        layout.addWidget(self.select_output_btn)

        self.output_dir_label = QLabel("No output directory selected")
        layout.addWidget(self.output_dir_label)

        # Excel file name input
        self.excel_name_label = QLabel("Output Excel File Name:")
        self.excel_name_input = QLineEdit("output_data.xlsx")
        layout.addWidget(self.excel_name_label)
        layout.addWidget(self.excel_name_input)

        # Progress display list widget
        self.progress_list = QListWidget()
        layout.addWidget(self.progress_list)

        # Run button
        self.run_btn = QPushButton("Run Workflow")
        self.run_btn.clicked.connect(self.run_workflow)
        layout.addWidget(self.run_btn)

        self.setLayout(layout)

    def select_files(self):
        # Allow the user to select multiple .tif files
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select .tif Images", "", "Image Files (*.tif)")
        
        if file_paths:
            self.file_paths = file_paths
            self.file_paths_label.setText(f"{len(file_paths)} files selected")
        else:
            self.file_paths_label.setText("No files selected")
            self.file_paths = None

    def select_output_directory(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_dir:
            self.output_dir_label.setText(output_dir)
            self.output_directory = output_dir
        else:
            self.output_dir_label.setText("No output directory selected")
            self.output_directory = None


    def run_workflow(self):
        if not hasattr(self, 'file_paths') or not self.file_paths:
            QMessageBox.warning(self, "Error", "Please select .tif images.")
            return

        if not hasattr(self, 'output_directory') or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        try:
            # Get user inputs
            gaussian_window = int(self.gaussian_window_input.text())
            gaussian_sigma = float(self.gaussian_sigma_input.text())
            dtime = float(self.dtime_input.text())
            conv_factor = dtime / 1E3

            # Get grid size and image size from the combo box
            selected_grid_image = self.grid_image_combo.currentText()
            grid_size, image_size = map(int, selected_grid_image.split(" - "))

            advanced_fitting = self.advanced_fitting_checkbox.isChecked()
            output_excel_file = os.path.join(self.output_directory, self.excel_name_input.text())

            # Loop through each selected .tif file
            for file_path in self.file_paths:
                image_name = os.path.splitext(os.path.basename(file_path))[0]
                logger.info(f"Processing file: {file_path}")
                start_time = time.time()
                
                try:
                    # Global analysis using GlobalImageProcessor
                    global_processor = GlobalImageProcessor(file_path, gaussian_window, gaussian_sigma, conv_factor, advanced_fitting)
                    raw_tif = global_processor.raw_frames
                    gaussian_frames, filtered_frames, _, autocorrelation_frames, results_frames, radial_autocorrelation_frames, radii = global_processor.process_frames()

                    AC_2D_models, numberofmolecules, waist_list, _, radial_fits = global_processor.radial(results_frames, radii)
                    global_stats = global_processor.compute_global_stats(numberofmolecules, waist_list, results_frames)

                    # Local analysis using LocalImageProcessor
                    local_processor = LocalImageProcessor(file_path, gaussian_window, gaussian_sigma, conv_factor, grid_size, image_size)
                    raw_subimage, filtered_subimage, normalized_subimage, number_of_molecules_array, debug_nfev_array, debug_success_array, debug_nit_array = local_processor.run()

                    mean_raw_subimage = np.nanmean(raw_subimage, axis=0)
                    mean_raw_subimage_array = np.array(mean_raw_subimage).reshape(grid_size, grid_size)
                    mean_filtered_subimage_array = np.nanmean(filtered_subimage, axis=0).reshape(grid_size, grid_size)
                    mean_normalized_subimage_array = np.nanmean(normalized_subimage, axis=0).reshape(grid_size, grid_size)
                    Brightness_grid = mean_raw_subimage_array / number_of_molecules_array
                    wait_mean = np.nanmean(waist_list)

                    plot_visualizations(
                        raw_tif, gaussian_frames, filtered_frames, autocorrelation_frames, AC_2D_models, 
                        radii, radial_autocorrelation_frames, radial_fits, mean_raw_subimage_array, 
                        mean_filtered_subimage_array, mean_normalized_subimage_array, number_of_molecules_array, 
                        Brightness_grid, debug_nfev_array, debug_success_array, debug_nit_array, wait_mean, self.output_directory, image_name
                    )

                    local_stats = {
                        'F': gaussian_window,
                        'S': gaussian_sigma,
                        'CR_moy': np.nanmean(mean_raw_subimage_array),
                        'CR_std': np.nanstd(mean_raw_subimage_array),
                        'CR_sem': np.nanstd(mean_raw_subimage_array) / np.sqrt(grid_size * grid_size),
                        'CRM_moy': np.nanmean(Brightness_grid),
                        'CRM_std': np.nanstd(Brightness_grid),
                        'CRM_sem': np.nanstd(Brightness_grid) / np.sqrt(grid_size * grid_size),
                        'N_moy': np.nanmean(number_of_molecules_array),
                        'N_std': np.nanstd(number_of_molecules_array),
                        'N_sem': np.nanstd(number_of_molecules_array) / np.sqrt(grid_size * grid_size),
                        'Conv': np.mean(debug_success_array)
                    }

                    excel_save(file_path, global_stats, local_stats, output_excel_file)
                    logger.info(f'Time taken for processing {file_path}: {time.time() - start_time} seconds')

                    # Update progress list
                    self.progress_list.addItem(f"Finished processing: {file_path}")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    self.progress_list.addItem(f"Error processing: {file_path}")
                    continue

            logger.info("All files processed.")
            QMessageBox.information(self, "Success", "Workflow completed successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Create tabs
        self.workflow_tab = WorkflowGUI()

        # Add tabs to the QTabWidget
        self.addTab(self.workflow_tab, "Image Processing Workflow")

        self.setWindowTitle("Image Processing")
        self.setGeometry(200, 200, 600, 600)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
