"""
PyRegX: Multivariate Regression Plugin for PyMOL (GUI Application)

This PyMOL plugin provides a graphical user interface (GUI) for performing Ordinary Least Squares (OLS) and Multiple Linear Regression (MLR) analyses on CSV-formatted datasets.
Users can load training and test datasets, specify the dependent variable, run the regression model, and export predictions and validation results.
The GUI includes built-in support for visual diagnostics such as correlation heatmaps and observed vs. predicted scatter plots.

Key Features:

             - Load training and test datasets (CSV)
             - Remove null-value columns
             - Define target (dependent) variable
             - Fit and validate OLS/MLR models using statsmodels and sklearn
             - Evaluate model using internal, external, and Leave-One-Out cross-validation metrics
             - Export results and predictions to .txt and .csv files
             - Generate correlation matrix and scatter plots
             - Integrated into PyMOL via plugin menu

Dependencies:
            - os, sys, subprocess, warnings
            - numpy
            - pandas
            - sklearn
            - statsmodels
            - seaborn, matplotlib
            - customtkinter
            - tkinter, tkinter.messagebox, tkinter.filedialog
            - pymol > v2.5, pymol.cmd

PyRegX Usage:
            1. Launch PyRegX GUI from PyMol: Plugin → PyRegX gui (v0.23)
            2. Load Training Dataset
                    o Click "Load Training Data" and select a .csv file
                    o The file must contain both independent and dependent variables
            3. Load Test Dataset
                    o Click "Load Test Data" and select a .csv file with the same structure
            4. Enter Dependent Variable (Y)
                    o Click "Enter Dependent Column"
                    o Provide the exact column name to be predicted
            5. Select Output File
                    o Click "Select Output File" to specify where to save results
            6. Run Analysis
                    o Click "Run Analysis"
                    o Generates regression summary, validation metrics, and prediction output
            7. Visual Output
                    o Correlation heatmap and scatter plot will pop up post-analysis

Author: Suvankar Banerjee
Email : suvankarbanerjee1995@gmail
Version: 0.23
"""

# Dependencies
import os
import sys
import subprocess
import importlib
import warnings
import customtkinter
import numpy as np
import pandas as pd
import sklearn
from tkinter import messagebox
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pymol
from pymol import cmd
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure required packages are installed
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    finally:
        globals()[import_name] = importlib.import_module(import_name)

# List of required packages
packages = {
    "customtkinter": "customtkinter",
    "pandas": "pandas",
    "numpy": "numpy",
    "statsmodels": "statsmodels",
    "sklearn": "sklearn",
    "pymol": "pymol",
}

for package, import_name in packages.items():
    install_and_import(package, import_name)

# Ignore statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# Class for MLR model validation
class OLS_MLR(object):
    
    y_name = None
    
    # class variables
    __version__ = "0.22"
    
    # define the init method
    def __init__(self):
        pass
        
    # import csv as dataframe
    def import_dataframe(self, path):
        try:
            return pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File '{path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    # remove null variable columns from the dataframes
    def remove_null_columns(self, data):
        # Get data
        self.df = data
        # Drop columns with null values
        self.clean_df = self.df.dropna(axis=1)
        return self.clean_df
    
    # process dataframe into X, Y, and Indexes
    def process_dataframe(self, data, y_name):
        # Get the data
        self.df = data
        # Add an index column by extracting the first column of the DataFrame
        self.index = self.df.index
        # Select the appropriate columns of the DataFrame
        self.x = self.df.drop(columns=[y_name])
        self.y = self.df[y_name]
        return self.index, self.x, self.y
    
    # Linear model with output
    def linear_model_analysis(self, train_index, x_train, y_train, test_index, x_test, Y_test):
        # Get train data
        self.train_index = train_index
        self.X_train = x_train
        self.Y_obs_train = y_train
        
        # Get test data
        self.test_index = test_index
        self.X_test = x_test
        self.Y_obs_test = Y_test
        
        # Calculate necessary variables
        self.N_train = len(self.X_train)
        self.N_test = len(self.X_test)
        self.P = len(self.X_train.T)
        
        # Add constant to X data
        self.X_train = statsmodels.api.add_constant(self.X_train, prepend=True)
        
        # Calculate Y mean values
        self.Y_obs_train_mean = np.mean(self.Y_obs_train)
        self.Y_obs_test_mean = np.mean(self.Y_obs_test)

        # Fit linear model using OLS regression
        self.lm = sm.OLS(self.Y_obs_train, self.X_train).fit()
        
        # Calculate predictions
        self.Y_pred_train = self.lm.predict(self.X_train)
        self.Y_pred_test = self.lm.predict(sm.add_constant(self.X_test, prepend=True))

        # Internal Validation
        self.PRESS = sum((self.Y_obs_train - self.Y_pred_train)**2)
        self.SEE = np.sqrt((sum((self.Y_obs_train - self.Y_pred_train)**2))/(self.N_train-self.P-1))
        self.MAE_train = sklearn.metrics.mean_absolute_error(self.Y_obs_train, self.Y_pred_train)
        self.MAE_test = sklearn.metrics.mean_absolute_error(self.Y_obs_test, self.Y_pred_test)
        
        # Calculate R2 values
        self.R2= sklearn.metrics.r2_score(self.Y_obs_train, self.Y_pred_train)
        self.R2_adjusted = self.lm.rsquared_adj
    
        # Parameters
        self.SSY = sum((self.Y_obs_train - self.Y_obs_test_mean)**2)
        self.VIF = 1/(1-self.R2)

        # External Validation (Without Scaling)
        self.RMSE_train = np.sqrt((sum((self.Y_obs_train - self.Y_pred_train)**2))/self.N_train)
        self.RMSE_pred_test = np.sqrt((sum((self.Y_obs_test - self.Y_pred_test)**2))/self.N_test)
        
        # Calculate R2 pred
        self.Q2f1 = 1-(sum((self.Y_obs_test - self.Y_pred_test)**2)/sum((self.Y_obs_test - self.Y_obs_train_mean)**2))
        self.Q2f2 = 1-(sum((self.Y_obs_test - self.Y_pred_test)**2)/sum((self.Y_obs_test - self.Y_obs_test_mean)**2))

        # Leave-One-Out(LOO) Cross Validation
        self.loo = linear_model.LinearRegression().fit(self.X_train, self.Y_obs_train)
        self.Y_pred_loo = sklearn.model_selection.cross_val_predict(self.loo, self.X_train, self.Y_obs_train, cv=self.N_train)
        self.Q2_loo = sklearn.metrics.r2_score(self.Y_obs_train, self.Y_pred_loo)
        self.MAE_loo = sklearn.metrics.mean_absolute_error(self.Y_obs_train, self.Y_pred_loo)
        
        # Summarize the results
        self.results = {
            "Summary": self.lm.summary(),
            "Internal Validation": {
                "SEE": round(self.SEE, 4),
                "MAE-Train": round(self.MAE_train, 5),
                "R2": round(self.R2, 4),
                "R2Adjusted": round(self.R2_adjusted, 4),
                "RMSE - Train": round(self.RMSE_train, 4),
                "SSY": round(self.SSY, 4),
                "VIF": round(self.VIF, 4),
                "PRESS": round(self.PRESS, 4)
            },
            "Leave-One-Out(LOO) Cross Validation": {
                "Q2 LOO": round(self.Q2_loo, 4),
                "MAE-LOO": round(self.MAE_loo, 4)
            },
            "External Validation (Without Scaling)": {
                "Q2f1 (R2 Pred)": round(self.Q2f1, 4),
                "Q2f2": round(self.Q2f2, 4),
                "RMSE-Test": round(self.RMSE_pred_test, 4),
                "MAE-Test": round(self.MAE_test, 4)
            }
        }
        
        return self.results, self.Y_pred_train, self.Y_pred_test


    #result Saver
    def results_to_text(self, results, path):
        
        # Get the results and filename 
        self.results = results
        
        # Get output path
        self.filename = path
        
        # Calling class variable using '__class__.variable'
        self.filepath = self.filename
        self.write = open(self.filepath, 'w')
        
        self.write.write("Summary:\n")
        self.write.write("==============================================================================")
        self.write.write(str(self.results["Summary"]))
        self.write.write(" \n")
        self.write.write("==============================================================================")
        
        self.write.write("\nInternal Validation:\n")
        for key, value in self.results["Internal Validation"].items():
            self.write.write(f"{key}: {value}\n")
        self.write.write("==============================================================================")
        
        self.write.write("\nLeave-One-Out(LOO) Cross Validation:\n")
        for key, value in self.results["Leave-One-Out(LOO) Cross Validation"].items():
            self.write.write(f"{key}: {value}\n")
        self.write.write("==============================================================================")
        
        self.write.write("\nExternal Validation (Without Scaling):\n")
        for key, value in self.results["External Validation (Without Scaling)"].items():
            self.write.write(f"{key}: {value}\n")
        self.write.write("==============================================================================")
        
        self.write.close()


    # scatter plot for Observed vs Predicted
    @staticmethod
    def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, alpha=1.0):
        plt.figure("Scatter plot")
        sns.scatterplot(x=y_train, y=y_pred_train, marker='o', color='blue', alpha=alpha, label='Training')
        sns.scatterplot(x=y_test, y=y_pred_test, marker='s', color='red', alpha=alpha, label='Test')
        sns.regplot(x=y_train, y=y_pred_train, scatter=False, color='black')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend(loc='lower right')
        plt.title('Actual vs Predicted Scatter Plot')
        plt.show()

    @staticmethod
    def plot_correlation_matrix(x, y):
        df = pd.concat([x, y], axis=1)
        corr = df.corr(method='pearson')
        plt.figure("Correlation Heatmap")
        sns.heatmap(corr, annot=True, fmt='.3f', cmap="coolwarm")
        plt.title('Pearson Correlation Matrix')
        plt.show()

# OLS GUI App
class PyRegX_gui(object):
    
    def __init__(self, master):
        # variables
        self.mlr = OLS_MLR()
        self.y_name = None
        self.results = None
        self.output_file = None
        self.path = None
        self.train_index = None
        self.x_train = None
        self.y_train = None
        self.test_index = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_test = None
        self.version = "0.23"
        
        # GUI
        self.master = master
        self.master.geometry("350x400")
        self.master.title("PyRegX gui v0.23")
        self.master.grid_columnconfigure(1, weight=2)
        self.master.grid_columnconfigure((2, 3), weight=0)
        self.master.grid_rowconfigure((0, 1, 2), weight=1)

        self.frame_1 = customtkinter.CTkFrame(master=self.master, bg_color='transparent',
                                              border_color="dodgerblue", fg_color=None)
        self.frame_1.pack(pady=10, padx=10, fill="both", expand=True)

        self.main()

    # main ui
    def main(self):
        # Create input fields
        self.train_data_input = customtkinter.CTkEntry(self.frame_1, width=250, fg_color="grey84", text_color='black')
        self.test_data_input = customtkinter.CTkEntry(self.frame_1, width=250, fg_color="grey84", text_color='black')
        self.output_file_input = customtkinter.CTkEntry(self.frame_1, width=250, fg_color="grey84", text_color='black')

        # Create labels for input fields
        self.label = customtkinter.CTkLabel(master=self.frame_1, text="PyRegX v0.23",
                                    font=customtkinter.CTkFont(size=15, weight="bold"))

        # Create buttons
        self.train_data_button = customtkinter.CTkButton(master=self.frame_1, text="Load Training Data",
                                                         fg_color="Dodgerblue3", command=self.load_train_data,
                                                         hover_color="SpringGreen4")
        
        self.test_data_button = customtkinter.CTkButton(master=self.frame_1, text="Load Test Data",
                                                        fg_color="Dodgerblue3", command=self.load_test_data,
                                                        hover_color="SpringGreen4")
        
        self.dialog_button = customtkinter.CTkButton(master=self.frame_1, text="Enter Dependant Column",
                                                       command=self.y_name_input_dialog,
                                                       fg_color="Dodgerblue3", hover_color="SpringGreen4")
        
        self.output_file_button = customtkinter.CTkButton(master=self.frame_1, text="Select Output File",
                                                          fg_color="Dodgerblue3", command=self.select_output_file,
                                                          hover_color="SpringGreen4")
        
        self.run_button = customtkinter.CTkButton(master=self.frame_1, text="Run Analysis", fg_color="firebrick3",
                                                  corner_radius=50, command=self.run_analysis, hover_color="SpringGreen4")
        
        # Add labels, input fields, and buttons to the frame
        self.label.pack(padx=1, pady=5)
        self.train_data_input.pack(padx=1, pady=1)
        self.train_data_button.pack(padx=1, pady=10)
        self.test_data_input.pack(padx=1, pady=1)
        self.test_data_button.pack(padx=1, pady=10)
        self.dialog_button.pack(padx=1, pady=10)
        self.output_file_input.pack(padx=1, pady=1)
        self.output_file_button.pack(padx=1, pady=10)
        self.run_button.pack(padx=10, pady=15)
        
    # Define a function to clear the Entry Widget Content
    def green_slate(self):
        # clear gui
        self.train_data_input.delete(0, 'end')
        self.test_data_input.delete(0, 'end')
        self.output_file_input.delete(0, 'end')
        
        # clear vars
        self.y_name = None
        self.results = None
        self.output_file = None
        self.train_index = None
        self.x_train = None
        self.y_train = None
        self.test_index = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_test = None

    # getting the conformation message box
    def success_msg(self):
        messagebox.showinfo("Status", "Job completed")
    
    # getting the conformation message box
    def error_msg(self):
        messagebox.showinfo("Status", "Error!")
    
    # function for training button
    def load_train_data(self):
        file_path = filedialog.askopenfilename(title="Load Training Data")
        if not file_path:
            return
        self.train_data_input.delete(0, 'end')
        self.train_data_input.insert(0, file_path)
        try:
            self.train_df = self.mlr.import_dataframe(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training data: {e}")

    # function for test button
    def load_test_data(self):
        # get file path
        file_path = filedialog.askopenfilename(title="Load Test Data")
        self.test_data_input.delete(0, 'end')
        self.test_data_input.insert(0, file_path)
        # load test data
        self.test_df = self.mlr.import_dataframe(file_path)

    # getting y data name from input dialog
    def y_name_input_dialog(self):
        dialog = customtkinter.CTkInputDialog(text="Enter Dependent Column:", title="Input Dialog")
        user_input = dialog.get_input()
        if not user_input or user_input not in self.train_df.columns:
            messagebox.showerror("Error", "Invalid column name. Please enter a valid dependent variable.")
            return None
        self.y_name = user_input

    # function for output button
    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(title="Select Output File")
        self.output_file_input.delete(0, 'end')
        self.output_file_input.insert(0, file_path)
        self.txt_path = (file_path + ".txt")
        self.path = file_path

    # function for run button
    def run_analysis(self):
        # Ensure the path is set
        if not self.path:
            messagebox.showerror("Error", "Output file path is not set. Please select an output file.")
            return
            
        # Ensure dataframes are loaded
        if self.train_df is None or self.test_df is None:
            messagebox.showerror("Error", "Training or testing data is missing.")
            return
            
        # Ensure y_name is set
        if self.y_name is None:
            messagebox.showerror("Error", "Target variable (y_name) is not specified.")
            return
            
        try:
            # Process data
            self.train_index, self.x_train, self.y_train = self.mlr.process_dataframe(self.train_df, self.y_name)
            self.test_index, self.x_test, self.y_test = self.mlr.process_dataframe(self.test_df, self.y_name)
            
            # Run OLS/MLR
            self.results, self.y_pred, self.y_pred_test = self.mlr.linear_model_analysis(self.train_index, self.x_train, self.y_train, self.test_index, self.x_test, self.y_test)
            
            # Save the results
            self.mlr.results_to_text(self.results, self.txt_path)
            
            # Save the output data
            train_mlr = pd.DataFrame({"index": self.train_index, "y_actual": self.y_train, "y_pred_MLR": self.y_pred})
            test_mlr = pd.DataFrame({"index": self.test_index, "y_actual": self.y_test, "y_pred_MLR": self.y_pred_test})
            
            train_csv_path = self.path + "_train.csv"
            test_csv_path = self.path + "_test.csv"
            
            train_mlr.to_csv(train_csv_path, index=False)
            test_mlr.to_csv(test_csv_path, index=False)
            
            # Show success message
            self.success_msg()
            
            # Generate plots
            self.mlr.plot_correlation_matrix(self.x_train, self.y_train)
            self.mlr.plot_scatter(self.y_train, self.y_pred, self.y_test, self.y_pred_test)
            
            # Clear UI elements
            self.green_slate()
        
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}")
            
            pass


def gui(self):
    root = customtkinter.CTk()
    app = PyRegX_gui(root)
    root.mainloop()

def __init__(self):
    self.menuBar.addmenuitem('Plugin', 'command',
                            'PyRegX gui v0.23',
                            label = 'PyRegX gui',
                            command = lambda : gui(self))


"""
LICENSE Info

Custom Non-Commercial License v1.0
Copyright © 2025 SUVANKAR BANERJEE <https://github.com/n0b0dy-95/DataPrep>
Email: suvankarbanerjee1995@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, merge, publish, and distribute the Software, subject to the following conditions:

    1. *Non-Commercial Use Only*  
       The Software may *only be used for non-commercial purposes*.  
       This means:
       - You may not sell, license, sublicense, or distribute this Software or any derivative works for a fee or other commercial benefit.
       - You may not use the Software as part of a commercial service, product, SaaS platform, or other revenue-generating activity.
       - You may not use the Software in any context where you, your company, or any third party derives a commercial advantage or financial compensation.

    2. *Attribution Required*  
       You must give appropriate credit to the original author(s), provide a link to this license, and indicate if changes were made.  
       Attribution must not suggest endorsement by the original author(s).

    3. *No Warranty*  
       The Software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.  
       In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from the use of the Software.

    4. *Redistribution*  
       You may redistribute this Software only if:
       - It is accompanied by this license in full.
       - It is clearly marked as a derivative (if modified).
       - It is not sold or used commercially.

    5. *Commercial Licensing Option*  
       To obtain a commercial license for use beyond these terms, including resale or integration into commercial offerings, please contact the author(s) at:  
       [suvankarbanerjee1995@gmail.com]

    6. *Termination*  
       Any violation of these terms automatically terminates your rights under this license.

---

By using the Software, you agree to the terms of this License.

This license shall be governed by Indian and international copyright laws, and any disputes arising under this license shall be resolved under the principles of fairness, good faith, and mutual respect, without limiting it to any single jurisdiction.

"""
