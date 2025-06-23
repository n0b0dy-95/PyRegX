# PyRegX: A Multivariate Regression Plug-in for PyMol

## Version:
    0.23 (for PyMol version > 2.5.0)

## Introduction
***PyRegX*** is a GUI-based plugin designed to perform robust ***Multiple Linear Regression (MLR)*** analysis on structured datasets directly from PyMol. Built with Customtkinter, it allows users to load datasets, define regression targets, validate models using statistical metrics, and export results-all without writing code.

## Key Features

      - Load training and test datasets in ***.CSV*** format
      - Automatic removal of null-valued columns
      - Select dependent (***Y***) variable *via* dialogbox
      - Run multivariate regression
      - **Validate models with**:
        - ***Internal Metrics:*** *R², Adjusted R², MAE, RMSE, VIF, PRESS, SEE*
        - External Metrics: Q²(f1, f2), MAE, RMSE
        - ***Cross-Validation:*** Leave-One-Out *Q²* and *MAE*
      - **Export**:
        - Statistical summaries (***.txt***)
        - Prediction tables (_train.csv, _test.csv)
      - **Visualization**:
        - Correlation Heatmap
        - Observed vs Predicted Scatter Plot
      - **Easy integration as a PyMol Plug-in** (Menu -> Plugin -> PyRegX gui)

# NOTE: The Index column must be at the first column of the input files.***

# Dependencies

Ensure the following packages are installed in your Python environment:

      - os, sys, subprocess, warnings
      - NumPy, Pandas
      - Scikit-Learn, statsmodels
      - Seaborn, Matplotlib
      - Tkinter
      - Customtkinter (for modern GUI components)
      - PyMol (for integration)

## Installation

   ### Option 1: Install in PyMol environment (manual)

      1. Place the plugin .py file in your PyMol plugins folder (e.g. ~/.pymol/startup/ or via Plugin Manager).
      2. Launch PyMol and navigate to **Plugin > Manage > Install** and select the .py file.
      3. PyRegX will appear in the **Plugin** menu.

   ###Option 2: Install required Python dependencies

      If you're not using a pre-configured Python environment, install dependencies with pip:

               $ pip install pandas numpy scikit-learn statsmodels matplotlib seaborn customtkinter

## Usage Workflow

      1. **Launch PyRegX GUI** from PyMol:
               *Plugin → PyRegX gui (v0.23)*
      2. **Load Training Dataset**
               A. Click *"Load Training Data"* and select a ***.csv*** file
               B. The file must contain both independent and dependent variables
      3. **Load Test Dataset**
               A. Click *"Load Test Data"* and select a .csv file with the same structure
      4. **Enter Dependent Variable (*Y*)**
               A. Click *"Enter Dependent Column"*
               B. Provide the exact column name to be predicted
      5. **Select Output File**
               A. Click *"Select Output File"* to specify where to save results
      6. **Run Analysis**
               A. Click *"Run Analysis"*
               B. Generates regression summary, validation metrics, and prediction output
      7. **Visual Output**
               A. Correlation heatmap and scatter plot will pop up post-analysis

# Author

**SUVANKAR BANERJEE**

**Email: <suvankarbanerjee1995@gmail.com>**

## LICENSE

**Custom Non-Commercial License v1.0**

Copyright © 2025 SUVANKAR BANERJEE <https://github.com/n0b0dy-95/DataPrep>

Email: suvankarbanerjee1995@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy

of this software and associated documentation files (the "Software"), to use,

copy, modify, merge, publish, and distribute the Software, subject to the following conditions:

***1. Non-Commercial Use Only***  

`   `The Software may \*only be used for non-commercial purposes\*.  

`   `This means:

`   `- You may not sell, license, sublicense, or distribute this Software or any derivative works for a fee or other commercial benefit.

`   `- You may not use the Software as part of a commercial service, product, SaaS platform, or other revenue-generating activity.

`   `- You may not use the Software in any context where you, your company, or any third party derives a commercial advantage or financial compensation.

***2. Attribution Required***  

`   `You must give appropriate credit to the original author(s), provide a link to this license, and indicate if changes were made.  

`   `Attribution must not suggest endorsement by the original author(s).

***3. No Warranty***  

`   `The Software is provided *"as is", without warranty of any kind*, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.  

`   `In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from the use of the Software.

***4. Redistribution***  

`   `You may redistribute this Software only if:

`   `- It is accompanied by this license in full.

`   `- It is clearly marked as a derivative (if modified).

`   `- It is not sold or used commercially.

***5. Commercial Licensing Option***  

`   `To obtain a commercial license for use beyond these terms, including resale or integration into commercial offerings, please contact the author(s) at: suvankarbanerjee1995@gmail.com

***6. Termination***  

`   `Any violation of these terms automatically terminates your rights under this license.

By using the Software, you agree to the terms of this License.

***This license shall be governed by Indian and international copyright laws, and any disputes arising under this license shall be resolved under the principles of fairness, good faith, and mutual respect, without limiting it to any single jurisdiction***.

