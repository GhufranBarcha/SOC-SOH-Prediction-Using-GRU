import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set the page configuration

st.set_page_config(layout="wide", page_title="SOC & SOH Prediction" ,page_icon="battery-half")
# st.set_page_config(layout="Centered", page_title="SOC & SOH Prediction",,page_icon="battery-half")
# Comment and uncomment above for different UI


font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)




# Load the trained GRU models
model_soc = tf.keras.models.load_model('gru_soc_model.h5')
model_soh = tf.keras.models.load_model('gru_soh_model.h5')

# Load the scaler parameters
scaler_params_soc = np.load('scaler_params_soc.npy', allow_pickle=True)
scaler_params_soh = np.load('scaler_params_soh.npy', allow_pickle=True)

# Define a function to perform SOC prediction
def SOC_pred(data):
    """
    Performs SOC prediction using a trained GRU model.

    Args:
        data: A Pandas DataFrame of SOC data.

    Returns:
        A Pandas DataFrame of predicted SOC values.
    """

    # Normalize the input data
    scaler = MinMaxScaler()
    scaler.min_ = scaler_params_soc[0]
    scaler.scale_ = scaler_params_soc[1]
    scaled_data = scaler.transform(data)

    # Reshape the input data to match the model's input shape
    time_steps = 100  # Should match the time_steps used during training
    n_features = scaled_data.shape[1]  # Should match the number of features in your input data
    input_data = []
    for i in range(len(scaled_data) - time_steps + 1):
        input_data.append(scaled_data[i : i + time_steps])
    input_data = np.array(input_data)

    # Perform prediction
    predicted_soc_scaled = model_soc.predict(input_data)

    # Reshape the predicted_soc_scaled array to match the original shape before inverse transformation
    predicted_soc_scaled_reshaped = predicted_soc_scaled.reshape(-1, 1)

    # Inverse transform the predicted SOC to get the original scale
    predicted_soc_original_scale = scaler.inverse_transform(predicted_soc_scaled_reshaped)

    return predicted_soc_original_scale

# Define a function to perform SOH prediction
def SOH_pred(data):
    """
    Performs SOH prediction using a trained GRU model.

    Args:
        data: A Pandas DataFrame of SOH data.

    Returns:
        A Pandas DataFrame of predicted SOH values.
    """

    # Normalize the input data
    scaler = MinMaxScaler()
    scaler.min_ = scaler_params_soh[0]
    scaler.scale_ = scaler_params_soh[1]
    scaled_data = scaler.transform(data)

    # Reshape the input data to match the model's input shape
    time_steps = 100  # Should match the time_steps used during training
    n_features = scaled_data.shape[1]  # Should match the number of features in your input data
    input_data = []
    for i in range(len(scaled_data) - time_steps + 1):
        input_data.append(scaled_data[i : i + time_steps])
    input_data = np.array(input_data)

    # Perform prediction
    predicted_soh_scaled = model_soh.predict(input_data)

    # Reshape the predicted_soh_scaled array to match the original shape before inverse transformation
    predicted_soh_scaled_reshaped = predicted_soh_scaled.reshape(-1, 1)

    # Inverse transform the predicted SOH to get the original scale
    predicted_soh_original_scale = scaler.inverse_transform(predicted_soh_scaled_reshaped)

    return predicted_soh_original_scale


st.title("GRU Prediction of SOC & SOH")
st.write(
    "Try uploading Excel file of last 100 data points to Predict for both SOH & SOC"
)



# Create tabs for SOC prediction, SOH prediction, and project info
tabs = st.tabs(["SOC Prediction", "SOH Prediction", "Project Info"])


# Create a page for SOC prediction
with tabs[0]:
    try:   # Read the SOC data from the uploaded file
        soc_upload = st.file_uploader("Upload an Excel File for SOC", type=["xlsx"])
        if soc_upload is not None:
            soc_data = pd.read_excel(soc_upload)

            # Perform SOC prediction
            predicted_soc = SOC_pred(soc_data)

            # Display the Pandas DataFrame of SOC data
            st.write("#### Input SOC Data ")
            st.dataframe(soc_data ,width= 2000)

            # Display the predicted SOC
            st.success('Predicted SOH : {:.3f}'.format(predicted_soc[0][0]))
    except Exception as e:
            st.warning('Shape of the uploaded file is incorrect. Please upload again.', icon="⚠️")    

# Create a page for SOH prediction
with tabs[1]:
    try: 
        # Read the SOH data from the uploaded file
        soh_upload = st.file_uploader("Upload an Excel File for SOH", type=["xlsx"])
        if soh_upload is not None:
            soh_data = pd.read_excel(soh_upload)

            # Perform SOH prediction
            predicted_soh = SOH_pred(soh_data)

            # Display the Pandas DataFrame of SOH data
            st.write("#### Input SOH Data ")
            st.dataframe(soh_data ,width= 2000)

            # Display the predicted SOH
            st.success('Predicted SOH : {:.3f}'.format(predicted_soh[0][0]))
    except Exception as e:
        st.warning('Shape of the uploaded file is incorrect. Please upload again.', icon="⚠️")



with tabs[2]:
    st.markdown(''' 
                
## Project Overview

This project focuses on building predictive models using Gated Recurrent Unit (GRU) neural networks to forecast both the State of Charge (SOC) and State of Health (SOH) in a battery system. Accurate SOC and SOH predictions are essential for battery management and energy optimization.

### Software Requirements

- Python 3 (higher than 3.5)
- Anaconda3 with Jupyter Notebook

### Getting Started

To run this project, follow these steps:

1. **Install Python**:
   - Make sure you have Python installed on your computer. You can download the latest version from the [official Python website](https://www.python.org/downloads/).
   - Jupyter Notebook comes pre-installed with the Anaconda distribution as well.

2. **Install Jupyter**:
   - You can install Jupyter Notebook using the following command in your terminal or command prompt:
     ```
     pip install jupyter
     ```

3. **Open Terminal/Command Prompt**:
   - Once you have Jupyter installed, open your terminal (Linux/Mac) or command prompt (Windows).

4. **Navigate to the Desired Directory**:
   - Use the `cd` command to navigate to the directory where you want to create or access your Jupyter Notebook files. For example:
     ```
     cd path/to/your/notebook/directory
     ```

5. **Start Jupyter Notebook**:
   - In the terminal/command prompt, enter the following command:
     ```
     jupyter notebook
     ```
   - This will start the Jupyter Notebook server and open a new tab or window in your default web browser.

### Project Workflow

#### Import Libraries

Importing necessary libraries is the initial step in any data analysis or machine learning project. In this case, libraries such as NumPy, Pandas, TensorFlow, Matplotlib, and MinMaxScaler are imported.

#### Load Data

Loading data is a fundamental step. The code loads data from an Excel file into a Pandas DataFrame. The dataset 'soc.xlsx' contains information about battery status.

#### Data Preprocessing

Data preprocessing is essential to ensure data quality and consistency. This includes:

- Checking for Null Values: `data.isnull().sum()` counts the number of missing values in the dataset, which helps identify and handle missing data.
- Checking for Duplicates: `data.duplicated().sum()` checks for and counts duplicated rows in the dataset, which can be removed if necessary.
- Selecting Columns: Specific columns relevant to SOC and SOH forecasting are selected for further analysis.
- Normalization: Data normalization is performed using Min-Max scaling (MinMaxScaler) to scale the data between 0 and 1, ensuring that all features have the same scale.

#### Data Visualization

Data visualization is an essential step to gain insights into the data. The code uses Matplotlib to plot the time series of 'MeasuredSOC' and 'MeasuredSOH,' which helps in understanding the data's behavior over time.

#### Define Time Steps and Features

In time series forecasting, data is typically divided into sequences. `time_steps` specifies the number of previous time steps to consider when predicting the next value. `n_features` is the number of features in the dataset, setting the stage for creating input sequences for the GRU models.

#### Prepare Dataset

To train GRU models, the data must be transformed into sequences of inputs (X) and corresponding targets (y). The code loops through the dataset to create overlapping sequences of time steps data points, which will be used to predict the next value (target).

#### Split Data

The dataset is divided into training and testing sets. The training set is used to train the models, while the testing set is used to evaluate their performance. The split is typically done by specifying a percentage, and in this case, 80% of the data is used for training.

#### Build the GRU Models

Gated Recurrent Unit (GRU) is used as the neural network architecture for sequence modeling. Two GRU models are created: one for SOC prediction and one for SOH prediction. The models consist of stacked GRU layers and a final Dense layer for regression. They are compiled with 'mean_squared_error' loss and 'Adam' optimizer.

#### Train the GRU Models

The models are trained on the training data using the `model.fit()` function. Training involves optimizing the models' parameters to minimize the specified loss function (mean squared error) over multiple iterations (epochs). A batch size of 64 is used, and the training progress is shown with `verbose=1`.

#### Predictions and RMSE

After training, the models make predictions on both the training and testing datasets. Root Mean Squared Error (RMSE) is calculated for both sets to quantify how well the models' predictions align with the actual values. RMSE is a common metric for regression tasks.

#### Plot the Predictions

The code creates visualizations to compare the true SOC and SOH values with the models' predictions for both the training and testing datasets. This step helps in visually assessing the models' performance.

#### Future SOC and SOH Prediction

The models are used to predict future SOC and SOH values based on the last 100 data points. This step demonstrates how the models can be applied to make forecasts beyond the original dataset.

#### Final Plot

The final plot combines the actual SOC and SOH values from the dataset with the predicted SOC and SOH values for visualization, giving a complete picture of the models' performance and their ability to forecast SOC and SOH values into the future.
   
                
                
                ''')

