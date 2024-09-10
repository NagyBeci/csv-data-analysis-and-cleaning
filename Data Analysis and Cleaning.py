import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import pdfkit
import os
from scipy import stats
import numpy as np
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define functions for data loading, cleaning, analysis, exporting, and reporting
def load_data(file_path):
    try:
        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            logging.error(f"The file {file_path} is empty.")
            return None

        data = pd.read_csv(file_path)

        # Check for a completely empty DataFrame
        if data.empty:
            logging.error(f"No data found in {file_path}.")
            return None

        logging.info(f"Data successfully loaded from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"No data in file: {file_path}")
        return None
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def clean_data(data):
    try:
        initial_row_count = data.shape[0]

        # Filter out rows that contain the word 'Total' in any column (case-insensitive)
        data = data[~data.apply(lambda row: row.astype(str).str.contains('Total', case=False).any(), axis=1)]

        # Remove leading and trailing spaces from string columns
        for col in data.select_dtypes(include=['object']):
            data.loc[:, col] = data[col].str.strip()

        # Remove currency symbols and other non-numeric characters from numeric columns
        for col in data.columns:
            if data[col].dtype == 'object':
                # Using .loc to ensure we're modifying the DataFrame in place
                data.loc[:, col] = data[col].replace('[\$,]', '', regex=True).str.strip()
                data.loc[:, col] = pd.to_numeric(data[col], errors='coerce')

        # Remove duplicates
        data = data.drop_duplicates()

        final_row_count = data.shape[0]
        logging.info(f"Cleaned data. Rows before: {initial_row_count}, Rows after: {final_row_count}")
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return None

def handle_outliers(data, column):
    """Handle outliers in a specific column using the Z-score method and log details."""
    original_data = data.copy()
    z_scores = stats.zscore(data[column])
    outliers_mask = (abs(z_scores) >= 3)
    outliers = data.loc[outliers_mask, column]

    # Log details of outliers
    for index, value in outliers.iteritems():
        logging.info(f"Outlier found in {column}: index={index}, value={value}")

    filtered_data = data.loc[~outliers_mask]
    logging.info(f"Outliers handled in {column}. Entries before: {len(data)}, Entries after: {len(filtered_data)}")
    return filtered_data

def normalize_data(data, column):
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    logging.info(f"Data normalized in column: {column}")
    return data

def impute_missing_values(data, column, method='mean'):
    if method == 'mean':
        data[column].fillna(data[column].mean(), inplace=True)
    elif method == 'median':
        data[column].fillna(data[column].median(), inplace=True)
    elif method == 'mode':
        data[column].fillna(data[column].mode()[0], inplace=True)
    logging.info(f"Missing values imputed in {column} using {method} method.")
    return data

def export_data(data, filename, format='csv'):
    try:
        if format == 'csv':
            data.to_csv(filename, index=False)
            logging.info(f"Data exported successfully in CSV format to {filename}")
        elif format == 'excel':
            data.to_excel(f"{filename}.xlsx", index=False)
            logging.info(f"Data exported successfully in Excel format to {filename}.xlsx")
        elif format == 'json':
            data.to_json(f"{filename}.json")
            logging.info(f"Data exported successfully in JSON format to {filename}.json")
        elif format == 'sql':
            from sqlalchemy import create_engine
            engine = create_engine('sqlite://', echo=False)
            data.to_sql(filename, con=engine)
            logging.info(f"Data exported successfully to SQL database {filename}")
        else:
            logging.error("Unsupported export format specified.")
    except Exception as e:
        logging.error(f"Error exporting data: {e}")

def export_all(data):
    # Ask user for directory to save files and option to create a new folder
    directory = filedialog.askdirectory()
    if directory:
        # Ask for a new folder name
        new_folder_name = simpledialog.askstring("Folder Name", "Enter a name for the new folder:")
        if new_folder_name:
            # Create a new folder within the selected directory
            new_folder_path = os.path.join(directory, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            # Generate and save all files into the new folder
            create_heatmap(data)
            shutil.move('heatmap.png', os.path.join(new_folder_path, 'heatmap.png'))

            create_pair_plot(data)
            shutil.move('pair_plot.png', os.path.join(new_folder_path, 'pair_plot.png'))

            create_histograms(data)
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                shutil.move(f"{col}_histogram.png", os.path.join(new_folder_path, f"{col}_histogram.png"))

            export_data(data, os.path.join(new_folder_path, 'analysis.csv'), 'csv')

            logging.info("All files exported successfully to " + new_folder_path)
            messagebox.showinfo("Export Successful", "All files have been exported to " + new_folder_path)
        else:
            logging.error("Export cancelled - no folder name provided.")
            messagebox.showwarning("Export Cancelled", "No folder name provided for export.")
    else:
        logging.error("Export cancelled.")
        messagebox.showwarning("Export Cancelled", "Export process was cancelled.")

def save_plots(data):
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{col}_histogram.png")
        plt.close()

def generate_report_with_visualizations(data, report_type='pdf'):
    save_plots(data)
    report_content = '<h1>Data Analysis Report</h1>'
    report_content += data.describe().to_html()
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        report_content += f'<h2>{col} Histogram</h2>'
        report_content += f'<img src="{col}_histogram.png">'
    if report_type == 'pdf':
        pdfkit.from_string(report_content, 'report_with_visualizations.pdf')
        logging.info("Automated report with visualizations generated in PDF format.")
    elif report_type == 'html':
        with open('report_with_visualizations.html', 'w') as file:
            file.write(report_content)
        logging.info("Automated report with visualizations generated in HTML format.")
    else:
        logging.error("Unsupported report format specified.")

def backup_data(file_path):
    backup_path = file_path + ".backup"
    try:
        shutil.copy(file_path, backup_path)
        logging.info(f"Backup created at {backup_path}")
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")

def upload_action():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = load_data(file_path)
        if data is not None:
            data = clean_data(data)
            messagebox.showinfo("Info", "Data loaded and cleaned successfully.")
        else:
            messagebox.showerror("Error", "Failed to load data.")

def export_action(export_type):
    export_path = filedialog.asksaveasfilename(defaultextension=f".{export_type}")
    if export_path and data is not None:
        if export_type == 'csv':
            export_data(data, export_path, 'csv')
        elif export_type == 'pdf':
            generate_report_with_visualizations(data, 'pdf')
            os.rename('report_with_visualizations.pdf', export_path)
        messagebox.showinfo("Info", f"Data exported successfully to {export_path}")
    elif not data:
        messagebox.showerror("Error", "No data to export.")

def create_heatmap(data):
    plt.figure()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Heatmap of Correlations")
    plt.savefig("heatmap.png")
    plt.close()
    logging.info("Heatmap created and saved.")

def create_pair_plot(data):
    plt.figure()
    sns.pairplot(data)
    plt.savefig("pair_plot.png")
    plt.close()
    logging.info("Pair plot created and saved.")

def create_histograms(data):
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{col}_histogram.png")
        plt.close()
    logging.info("Histograms created and saved.")

def export_all(data):
    # Ask user for directory to save files
    directory = filedialog.askdirectory()
    if directory:
        # Ask for a new folder name
        new_folder_name = simpledialog.askstring("Folder Name", "Enter a name for the new folder:")
        if new_folder_name:
            # Create a new folder within the selected directory
            new_folder_path = os.path.join(directory, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            # Generate and move all files into the new folder
            create_heatmap(data)
            shutil.move('heatmap.png', os.path.join(new_folder_path, 'heatmap.png'))

            create_pair_plot(data)
            shutil.move('pair_plot.png', os.path.join(new_folder_path, 'pair_plot.png'))

            create_histograms(data)
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                shutil.move(f"{col}_histogram.png", os.path.join(new_folder_path, f"{col}_histogram.png"))

            export_data(data, os.path.join(new_folder_path, 'analysis.csv'), 'csv')

            logging.info("All files exported successfully to " + new_folder_path)
            messagebox.showinfo("Export Successful", "All files have been exported to " + new_folder_path)
        else:
            logging.error("Export cancelled - no folder name provided.")
            messagebox.showwarning("Export Cancelled", "No folder name provided for export.")
    else:
        logging.error("Export cancelled.")
        messagebox.showwarning("Export Cancelled", "Export process was cancelled.")

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)

def create_ui():
    window = tk.Tk()
    window.title("Data Analysis and Cleaning Tool")
    window.geometry("600x400")

    main_frame = tk.Frame(window, padx=10, pady=10)
    main_frame.pack(expand=True, fill=tk.BOTH)

    # Frame for Upload and Export All buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill=tk.X)

    # Upload CSV Button
    upload_button = tk.Button(button_frame, text="Upload CSV", command=upload_action)
    upload_button.pack(side=tk.LEFT, expand=True)

    # Export All Button
    export_all_button = tk.Button(button_frame, text="Export All", command=lambda: export_all(data))
    export_all_button.pack(side=tk.LEFT, expand=True)

    # Export All Analysis Button
    export_all_button = tk.Button(button_frame, text="Export All Analysis", command=lambda: export_all(data))
    export_all_button.pack(side=tk.LEFT, expand=True)

    # Progress Label
    progress_label = tk.Label(main_frame, text="")
    progress_label.pack()

    # Logging Window setup
    bottom_frame = tk.Frame(main_frame)
    bottom_frame.pack(expand=True, fill=tk.BOTH)

    log_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=10, state='disabled')
    log_text.pack(fill=tk.BOTH, expand=True)

    # Configure logging to display in the scrolledtext widget
    text_handler = TextHandler(log_text)
    logging.getLogger().addHandler(text_handler)
    logging.getLogger().setLevel(logging.INFO)

    window.mainloop()

# Main program execution
if __name__ == "__main__":
    create_ui()
