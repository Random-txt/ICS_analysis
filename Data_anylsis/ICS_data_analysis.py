import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from sklearn.metrics import r2_score
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# Small epsilon value to avoid divisions by zero
epsilon = 1e-10

# Function to load the Excel file
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        required_columns = ['CRmoy', 'CRMmoy', 'CRMsem']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error loading data: {str(e)}")
        return None

# Model functions
def model_function_simple(CRmoy, CRM0, Ssm, CR1moy):
    return CRM0 * (1 + Ssm * CRmoy / CR1moy)

def model_function_BG(CRmoy, CRM0, Ssm, BG, CR1moy):
    p = CRmoy / (CR1moy + epsilon)
    r_BG = BG / (CR1moy + epsilon)
    return CRM0 * (1 - r_BG / (p + epsilon)) * (1 + Ssm * (p - r_BG) / (1 - r_BG + epsilon))

# Function to fit the data for simple model
def fit_simple_model(CRmoy, CRMmoy, CR1moy, bounds):
    initial_guess_simple = [1.0, 1.0]
    popt_simple, pcov_simple = curve_fit(lambda CRmoy, CRM0, Ssm: model_function_simple(CRmoy, CRM0, Ssm, CR1moy), 
                                         CRmoy, CRMmoy, p0=initial_guess_simple, bounds=bounds)
    return popt_simple, pcov_simple

# Function to fit the data for background model
def fit_bg_model(CRmoy, CRMmoy, CR1moy, bounds):
    initial_guess_BG = [1.0, 1.0, 1.0]
    popt_BG, pcov_BG = curve_fit(lambda CRmoy, CRM0, Ssm, BG: model_function_BG(CRmoy, CRM0, Ssm, BG, CR1moy), 
                                 CRmoy, CRMmoy, p0=initial_guess_BG, bounds=bounds)
    return popt_BG, pcov_BG

# Function to calculate additional columns and export results
def export_results(df, popt, std_devs, CRM0_fit, SSm_fit, short_file_name, molecule_name, user_defined, file_output_path, experiment_date, comment, DegLab):
    rows = []

    result = {
        'Experiment date': experiment_date,
        'Molecule': molecule_name,  # Use the provided molecule information
        'Mass factor': user_defined,
        'File/ref.': short_file_name,
        'DegLab': DegLab,  # User-defined DegLab
        'Nhv': np.nan,  # Placeholder, update if necessary
        'varNhv': np.nan,  # Placeholder, update if necessary
        'Nhv+Nhv^2/Nmolec': np.nan,  # Placeholder, update if necessary
        'Nmolec': df['Nmoy'].iloc[0],
        'dNmolec': df['Nstd'].iloc[0],
        'CV': df['Nstd'].iloc[0] / df['Nmoy'].iloc[0],
        'Plas': np.nan,  # Placeholder, update if necessary
        'EpsP1': np.nan,  # Placeholder, update if necessary
        'Comment': comment,  # User-defined comment for the analysis
        'wr': np.mean(df['waist']),
        'CR1moy': df['CRmoy'].iloc[0],
        'CRM0': CRM0_fit,
        'dCRM0': std_devs[0],
        'Ssm': SSm_fit,
        'dSsm': std_devs[1]
    }
    
    # Calculate additional values
    result['cFluo'] = result['CR1moy'] / result['CRM0'] / result['DegLab'] / np.pi / result['wr']**2
    result['minf'] = 1 / (1 - 3 / 8 * result['Ssm'])
    result['msup'] = 2 / (2 - result['Ssm']) if result['Ssm'] < 1 else 3 / (2 - result['Ssm']/2)
    result['p1min'] = 3 - result['msup'] * (2 - 0.5 * result['Ssm'])
    result['p1max'] = 3 - result['minf'] * (2 - 0.5 * result['Ssm'])
    result['p2min'] = -3 + result['minf'] * (3 - result['Ssm'])
    result['p2max'] = -3 + result['msup'] * (3 - result['Ssm'])
    result['p3min'] = 1 - 0.5 * result['msup'] * (2 - result['Ssm'])
    result['p3max'] = 1 - 0.5 * result['minf'] * (2 - result['Ssm'])
    result['cMassMin'] = result['cFluo'] / result['msup'] * result['Mass factor']
    result['cMassMax'] = result['cFluo'] / result['minf'] * result['Mass factor']
    
    rows.append(result)
    
    # Check if the file exists and append data
    if os.path.exists(file_output_path):
        # Load the existing Excel file and append the new data
        existing_df = pd.read_excel(file_output_path)
        export_df = pd.DataFrame(rows)
        updated_df = pd.concat([existing_df, export_df], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        updated_df = pd.DataFrame(rows)

    # Save the updated DataFrame back to the Excel file
    updated_df.to_excel(file_output_path, index=False)
    print(f"Data appended and exported successfully to {file_output_path}")

# Function to calculate R² and chi²
def calculate_statistics(CRmoy, CRMmoy, model_moy, CRMsem):
    residuals = CRMmoy - model_moy
    chi_squared = np.sum((residuals / CRMsem) ** 2)
    r_squared = r2_score(CRMmoy, model_moy)
    return r_squared, chi_squared

# Function to plot results in the GUI
def plot_results(canvas, figure, ax, df, excluded_df, popt, model_type, CR1moy, plot_color):
    CRmoy = df['CRmoy'].values
    CRMmoy = df['CRMmoy'].values
    CRMsem = df['CRMsem'].values

    # Clear the previous plot
    ax.clear()

    # Generate fitted model data
    if model_type == 'simple':
        CRMmoy_fitted = model_function_simple(CRmoy, *popt, CR1moy)
    else:
        CRMmoy_fitted = model_function_BG(CRmoy, *popt, CR1moy)

    # Plot the original data
    ax.errorbar(CRmoy, CRMmoy, yerr=CRMsem, fmt='o', label='All Data', color='blue', alpha=0.5)

    # Plot excluded data separately
    if not excluded_df.empty:
        ax.scatter(excluded_df['CRmoy'], excluded_df['CRMmoy'], label='Excluded Data', color='red', marker='x')

    # Plot fitted curve
    ax.plot(CRmoy, CRMmoy_fitted, label='Fitted Curve', color=plot_color)

    # Calculate R² and chi²
    r_squared, chi_squared = calculate_statistics(CRmoy, CRMmoy, CRMmoy_fitted, CRMsem)

    # Show statistics and parameters in legend
    parameters_text = f"R² = {r_squared:.3f}, χ² = {chi_squared:.3f}\nCRM0 = {popt[0]:.3f}, Ssm = {popt[1]:.3f}"
    if model_type == 'bg':
        parameters_text += f", BG = {popt[2]:.3f}"
    ax.legend([parameters_text])

    # Set labels
    ax.set_xlabel('CRmoy (kHz)')
    ax.set_ylabel('CRMmoy (kHz/mol)')
    ax.set_title(f'Fitted Curves vs Original Data ({model_type.title()})')

    # Redraw the canvas
    canvas.draw()

# Function to save the plot as an image with higher DPI
def save_plot(figure, dpi=300):
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
    if file_path:
        figure.savefig(file_path, dpi=dpi)  # Set the desired DPI during saving
        messagebox.showinfo("Success", f"Plot saved successfully to {file_path}")


# Main GUI setup
def main_gui():
    def load_file():
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            df = load_data(file_path)
            if df is not None:
                global CR1moy, data_df, short_file_name
                data_df = df
                CR1moy = df['CRmoy'].values[0]  # First value of CRmoy
                short_file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract file name without extension
                file_name_var.set(short_file_name)
                messagebox.showinfo("Info", f"Data loaded successfully from {file_path}")

    def choose_color():
        color = colorchooser.askcolor()[1]
        if color:
            color_var.set(color)

    def select_output_file():
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if output_path:
            output_file_var.set(output_path)

    def run_analysis():
        if data_df is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        try:
            model_choice = model_var.get()
            CRM0_lower = float(CRM0_lower_entry.get())
            CRM0_upper = float(CRM0_upper_entry.get())
            Ssm_lower = float(Ssm_lower_entry.get())
            Ssm_upper = float(Ssm_upper_entry.get())

            # Get and validate DegLab value
            DegLab = float(DegLab_entry.get())
            if DegLab < 0 or DegLab > 1:
                messagebox.showerror("Error", "DegLab must be between 0 and 1.")
                return

            # Exclude data points
            exclusion_indices = exclusion_entry.get().strip()
            if exclusion_indices:
                exclude_indices = list(map(int, exclusion_indices.split(',')))
                excluded_df = data_df.iloc[exclude_indices]
                df_for_fitting = data_df.drop(exclude_indices).reset_index(drop=True)
            else:
                excluded_df = pd.DataFrame()  # No rows excluded
                df_for_fitting = data_df

            if model_choice == 'bg':
                BG_lower = float(BG_lower_entry.get())
                BG_upper = float(BG_upper_entry.get())
                bounds = ([CRM0_lower, Ssm_lower, BG_lower], [CRM0_upper, Ssm_upper, BG_upper])
                popt, pcov = fit_bg_model(df_for_fitting['CRmoy'].values, df_for_fitting['CRMmoy'].values, CR1moy, bounds)
            else:
                bounds = ([CRM0_lower, Ssm_lower], [CRM0_upper, Ssm_upper])
                popt, pcov = fit_simple_model(df_for_fitting['CRmoy'].values, df_for_fitting['CRMmoy'].values, CR1moy, bounds)

            std_devs = np.sqrt(np.diag(pcov))
            output_file = output_file_var.get() or "exported_data.xlsx"
            molecule_name = molecule_entry.get() or "Unknown"
            comment = comment_entry.get() or ""
            # Get experiment date or use the current date
            experiment_date = experiment_date_entry.get()
            if not experiment_date:
                experiment_date = pd.Timestamp.now().isoformat()
            export_results(df_for_fitting, popt, std_devs, popt[0], popt[1], short_file_name, molecule_name, 1.0, output_file, experiment_date, comment, DegLab)
            plot_results(canvas, figure, ax, df_for_fitting, excluded_df, popt, model_choice, CR1moy, color_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    root = tk.Tk()
    root.title("Data Analysis Tool")

    # Organize widgets in a grid layout, leaving space for the plot on the right side
    control_frame = tk.Frame(root)
    control_frame.grid(row=0, column=0, padx=10, pady=10)

    load_button = tk.Button(control_frame, text="Load Data File", command=load_file)
    load_button.grid(row=0, column=0, padx=10, pady=10)

    model_var = tk.StringVar(value='simple')
    tk.Radiobutton(control_frame, text="Simple Model", variable=model_var, value='simple').grid(row=1, column=0, padx=10, pady=5)
    tk.Radiobutton(control_frame, text="Model with BG", variable=model_var, value='bg').grid(row=1, column=1, padx=10, pady=5)

    tk.Label(control_frame, text="CRM0 Lower Bound:").grid(row=2, column=0, padx=10, pady=5)
    CRM0_lower_entry = tk.Entry(control_frame)
    CRM0_lower_entry.grid(row=2, column=1, padx=10, pady=5)
    CRM0_lower_entry.insert(0, "0")

    tk.Label(control_frame, text="CRM0 Upper Bound:").grid(row=3, column=0, padx=10, pady=5)
    CRM0_upper_entry = tk.Entry(control_frame)
    CRM0_upper_entry.grid(row=3, column=1, padx=10, pady=5)
    CRM0_upper_entry.insert(0, "inf")

    tk.Label(control_frame, text="Ssm Lower Bound:").grid(row=4, column=0, padx=10, pady=5)
    Ssm_lower_entry = tk.Entry(control_frame)
    Ssm_lower_entry.grid(row=4, column=1, padx=10, pady=5)
    Ssm_lower_entry.insert(0, "0")

    tk.Label(control_frame, text="Ssm Upper Bound:").grid(row=5, column=0, padx=10, pady=5)
    Ssm_upper_entry = tk.Entry(control_frame)
    Ssm_upper_entry.grid(row=5, column=1, padx=10, pady=5)
    Ssm_upper_entry.insert(0, "inf")

    tk.Label(control_frame, text="BG Lower Bound:").grid(row=6, column=0, padx=10, pady=5)
    BG_lower_entry = tk.Entry(control_frame)
    BG_lower_entry.grid(row=6, column=1, padx=10, pady=5)
    BG_lower_entry.insert(0, "0")

    tk.Label(control_frame, text="BG Upper Bound:").grid(row=7, column=0, padx=10, pady=5)
    BG_upper_entry = tk.Entry(control_frame)
    BG_upper_entry.grid(row=7, column=1, padx=10, pady=5)
    BG_upper_entry.insert(0, "inf")

    tk.Label(control_frame, text="Exclude Data Points (comma-separated indices):").grid(row=8, column=0, padx=10, pady=5)
    exclusion_entry = tk.Entry(control_frame)
    exclusion_entry.grid(row=8, column=1, padx=10, pady=5)

    tk.Label(control_frame, text="DegLab (Float:[0-1]):").grid(row=9, column=0, padx=10, pady=5)
    DegLab_entry = tk.Entry(control_frame)
    DegLab_entry.grid(row=9, column=1, padx=10, pady=5)
    DegLab_entry.insert(0, "1")

    color_var = tk.StringVar(value='green')
    color_button = tk.Button(control_frame, text="Choose Plot Color", command=choose_color)
    color_button.grid(row=10, column=0, padx=10, pady=5)

    output_file_var = tk.StringVar(value="")
    output_file_button = tk.Button(control_frame, text="Select Output File", command=select_output_file)
    output_file_button.grid(row=11, column=0, padx=10, pady=5)

    tk.Label(control_frame, text="Molecule Name:").grid(row=12, column=0, padx=10, pady=5)
    molecule_entry = tk.Entry(control_frame)
    molecule_entry.grid(row=12, column=1, padx=10, pady=5)

    tk.Label(control_frame, text="Short File Name:").grid(row=13, column=0, padx=10, pady=5)
    file_name_var = tk.StringVar()
    file_name_entry = tk.Entry(control_frame, textvariable=file_name_var, state="readonly")
    file_name_entry.grid(row=13, column=1, padx=10, pady=5)

    tk.Label(control_frame, text="Experiment Date (YYYY-MM-DD):").grid(row=14, column=0, padx=10, pady=5)
    experiment_date_entry = tk.Entry(control_frame)
    experiment_date_entry.grid(row=14, column=1, padx=10, pady=5)

    tk.Label(control_frame, text="Comment:").grid(row=15, column=0, padx=10, pady=5)
    comment_entry = tk.Entry(control_frame)
    comment_entry.grid(row=15, column=1, padx=10, pady=5)

    analyze_button = tk.Button(control_frame, text="Run Analysis", command=run_analysis)
    analyze_button.grid(row=16, column=0, columnspan=2, pady=20)

    save_plot_button = tk.Button(control_frame, text="Save Plot", command=lambda: save_plot(figure))
    save_plot_button.grid(row=17, column=0, columnspan=2, pady=10)

    # Create the plot area on the right side
    figure, ax = plt.subplots(figsize=(5, 4))
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().grid(row=0, column=1, rowspan=18, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    data_df = None
    short_file_name = ""
    main_gui()
