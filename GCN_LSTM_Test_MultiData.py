import os
import sys
import urllib.request

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os
import statistics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import GRU, SimpleRNN, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD,Adam,RMSprop

from sklearn.model_selection import KFold
from stellargraph.layer import GCN_LSTM
import optuna
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Set random seed for Python's built-in random module
import random
random.seed(42)

import stellargraph as sg
import sys

## Check if we are using GPU or CPU
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))




try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None


# Set random seed for TensorFlow
tf.random.set_seed(42)
# Set the random seed for reproducibility
np.random.seed(42)


"""## Data

We apply the GCN-LSTM model to the **Los-loop** data. This traffic dataset
contains traffic information collected from loop detectors in the highway of Los Angeles County (Jagadish
et al., 2014).  There are several processed versions of this dataset used by the research community working in Traffic forecasting space.

This demo is based on the preprocessed version of the dataset used by the TGCN paper. It can be directly accessed from there [github repo](https://github.com/lehaifeng/T-GCN/tree/master/data).

This dataset  contains traffic speeds from Mar.1 to Mar.7, 2012 of 207 sensors, recorded every 5 minutes.

In order to use the model, we need:

* A N by N adjacency matrix, which describes the distance relationship between the N sensors,
* A N by T feature matrix, which describes the (f_1, .., f_T) speed records over T timesteps for the N sensors.

A couple of other references for the same data albeit different time length are as follows:

* [DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING](https://github.com/liyaguang/DCRNN/tree/master/data): This dataset consists of 207 sensors and collect 4 months of data ranging from Mar 1st 2012 to Jun 30th 2012 for the experiment. It has some missing values.
* [ST-MetaNet: Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning](https://github.com/panzheyi/ST-MetaNet/tree/master/traffic-prediction). This work uses the DCRNN preprocessed data.

## Loading and preprocessing the data
"""


"""This demo is based on the preprocessed version of the dataset used by the TGCN paper."""

# Number of data points
num_points = 690
#Num_Dat = 31
Num_St = 98
#Pred_line = 0
epc_number=2000
train_rate = 0.833
seq_len = 20
pre_len = 0
corr_threshold = 0.7 #Thresold for correlation consideration
distance_threshold = 5000  # Adjust this value based on your specific dataset
# Set the window size for the moving average
window_size = 10
#Success_threshold = 0.6

# Read the Excel file for training and validation
file_path1 = r'C:\UConn\Ethiopia\GCNLSTM_ERA5_CMIP\SC1\ERA5_Land_Hist.xlsx'
# Read the Excel file for testing or forecasting
file_path2 = r'C:\UConn\Ethiopia\GCNLSTM_ERA5_CMIP\SC1\ERA5_Land_Hist.xlsx'
dataaa = pd.read_excel(file_path1, header=None)

# Optionally, print the filtered DataFrame
#print('print of Input Data: ',dataaa)

dataaa_F = pd.read_excel(file_path2)
dataaa_F = dataaa_F.T

Input = np.array(dataaa)
Input_F = np.array(dataaa_F)

num_nodes, time_len = Input.shape
#print("No. of Stations:", num_nodes, "\nNo of timesteps:", time_len)
          

# Define the number of folds
k = 6
kf = KFold(n_splits=k)

# starts from 0 to 5 (Means 1 - 6):
desired_fold = 0

scores = []

# K-Fold Cross-Validation
kk=0
for ki, (train_index, val_index) in enumerate(kf.split(Input)):
  print (ki)  
  if ki == desired_fold:  
    # Create training and validation sets for this fold
    train_data, test_data = Input[train_index], Input[val_index]
    train_data = train_data.T
    test_data = test_data.T

    #train_data, test_data = train_test_split(Input, train_rate)
    print("Train data: ", train_data.shape)
    print("Test data: ", test_data.shape)

    x = np.linspace(0, train_data.shape[1], train_data.shape[1])  # Example x values
    y = train_data  # Example y values

    # Plot the NumPy array, to show the objective you need to active the next 5 lines:
    plt.plot(x, y[92])
    plt.xlabel('Woreda')
    plt.ylabel('Yeild')
    plt.title('Plot of Yeilding')
    plt.show()

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

    # Initialize the StandardScaler
    #scaler = RobustScaler()    
    #scaler = StandardScaler()
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and validation sets
    train_scaled = scaler.fit_transform(train_data.T)
    test_scaled = scaler.transform(test_data.T)
    F_scaled = scaler.transform(Input_F.T)
    train_scaled = train_scaled.T
    test_scaled = test_scaled.T
    F_scaled = F_scaled.T

    print("Train scaled shape: ", train_scaled.shape)
    print("Test scaled shape: ", test_scaled.shape)
    print("Forecast scaled shape: ", F_scaled.shape)


    def sequence_data_preparation(seq_len, pre_len, train_data, test_data, F_scaled):
        trainX, trainY, testX, testY, ForecastX, ForecastY = [], [], [], [], [], []

        for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
            a = train_data[:, i : i + seq_len + pre_len]
            trainX.append(a[:, :seq_len])
            trainY.append(a[:, -1])

        for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
            b = test_data[:, i : i + seq_len + pre_len]
            testX.append(b[:, :seq_len])
            testY.append(b[:, -1])

        for i in range(F_scaled.shape[1] - int(seq_len + pre_len - 1)):
            F = F_scaled[:, i : i + seq_len + pre_len]
            ForecastX.append(F[:, :seq_len])
            ForecastY.append(F[:, -1])

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        ForecastX = np.array(ForecastX)
        ForecastY = np.array(ForecastY)

        return trainX, trainY, testX, testY, ForecastX, ForecastY

    trainX, trainY, testX, testY, ForecastX, ForecastY = sequence_data_preparation(seq_len, pre_len, train_scaled, test_scaled, F_scaled)

    print("Train X scaled shape: ", trainX.shape)
    print("Train Y scaled shape: ", trainY.shape)
    print("Test X scaled shape: ", testX.shape)
    print("Test Y scaled shape: ", testY.shape)
    print("Forecast X scaled shape: ", ForecastX.shape)
    print("Forecast Y scaled shape: ", ForecastY.shape)


    #### Configure the Graph Matrix:
    import torch
    from scipy.spatial.distance import cdist
    torch.manual_seed(42)

    #we need to calculate the correlation matrix:
    Correlation_Matrix = np.corrcoef(train_data.T, rowvar=False)
    np.fill_diagonal(Correlation_Matrix, 0)

    # Find the index of the maximum correlation in each row
    max_correlation_index = np.argmax(Correlation_Matrix, axis=1)

    adj_matrix = torch.tensor(abs(Correlation_Matrix) >= corr_threshold, dtype=torch.float)

    for i in range(Num_St):
        adj_matrix[i,max_correlation_index[i]] = 1
        adj_matrix[max_correlation_index[i],i] = 1

    # Set the diagonal elements to zero (assuming self-loops are not allowed)
    #adj_matrix.fill_diagonal_(0)

    # Print the adjacency matrix
    print('Adjacancy matrix shape: ',adj_matrix.shape)

    import networkx as nx
    import matplotlib.pyplot as plt
    import torch
    from scipy.spatial.distance import cdist

    # Convert the adjacency matrix to a NetworkX graph
    graph = nx.from_numpy_array(adj_matrix.numpy())

    # Plot the graph, To show the Graph you need to active the next two lines:
    #nx.draw(graph, with_labels=True)
    #plt.show()


    ########   ******* Hyper parameters tunning: *******************************
    study = None  # Define study in a broader scope
    def create_model(trial):
        # Suggest the number of layers for GCN and LSTM
        num_gc_layers = trial.suggest_int("num_gc_layers", 1, 3)  # Adjust range as needed
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)  # Adjust range as needed

        # Suggest sizes for each GCN layer
        gc_layer_sizes = [trial.suggest_int(f"gc_layer_{i}", 8, 256) for i in range(num_gc_layers)]
    
        # Suggest sizes for each LSTM layer
        lstm_layer_sizes = [trial.suggest_int(f"lstm_layer_{i}", 100, 400) for i in range(num_lstm_layers)]

        # Define possible activation functions
        activation_functions = ["relu", "softsign", "tanh", "elu", "swish", "gelu"]

        # Suggest activation functions for GCN layers
        gc_activations = [trial.suggest_categorical(f"gc_activation_{i}", activation_functions) for i in range(num_gc_layers)]
    
        # Suggest activation functions for LSTM layers
        lstm_activations = [trial.suggest_categorical(f"lstm_activation_{i}", activation_functions) for i in range(num_lstm_layers)]

        # Suggest learning rate
        learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)

        # Define possible loss functions to choose from
        loss_functions = ["mse", "mae", "huber_loss"]  # You can add more loss functions if needed
        loss_function = trial.suggest_categorical("loss_function", loss_functions)

        # Define possible optimizers to choose from
        optimizers = ["adam", "rmsprop"]  # You can add other optimizers like Adagrad, Adamax, etc.
        optimizer_name = trial.suggest_categorical("optimizer", optimizers)

        # Configure optimizer with associated parameters
        if optimizer_name == "adam":
            beta_1 = trial.suggest_float("adam_beta_1", 0.8, 0.999)
            beta_2 = trial.suggest_float("adam_beta_2", 0.9, 0.999)
            optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        elif optimizer_name == "sgd":
            momentum = trial.suggest_float("sgd_momentum", 0.0, 0.9)
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer_name == "rmsprop":
            rho = trial.suggest_float("rmsprop_rho", 0.85, 0.99)
            optimizer = RMSprop(learning_rate=learning_rate, rho=rho)

        # Suggest batch size
        batch_size = trial.suggest_int("batch_size", 16, 128)  # Define the range for batch size

        # Suggest dropout rate
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  # Adjust range for dropout rate


        # Initialize GCN_LSTM with trial parameters
        gcn_lstm = GCN_LSTM(
            seq_len=seq_len,
            adj=adj_matrix,
            gc_layer_sizes=gc_layer_sizes,
            gc_activations=gc_activations,
            lstm_layer_sizes=lstm_layer_sizes,
            lstm_activations=lstm_activations,
            dropout=dropout,  # Change this to your desired dropout value
        )

        x_input, x_output = gcn_lstm.in_out_tensors()
        model = Model(inputs=x_input, outputs=x_output)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Train your model
        history = model.fit(
            trainX,
            trainY,
            epochs=50,
            batch_size=batch_size, # Use the tuned batch size
            shuffle=False,
            verbose=1,
            validation_data=(testX, testY),
            callbacks=[early_stopping]
        )

        # Return the best validation loss
        return min(history.history['val_loss'])

    def optimize():
        global study  # Use global to modify the study variable
        study = optuna.create_study(direction="minimize")
        study.optimize(create_model, n_trials=100)  # Adjust the number of trials as needed

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    if __name__ == "__main__":
        # Optimize and get best trial object
        optimize()
        # After optimization, use best parameters to create final model
        best_trial = study.best_trial


        gcn_lstm = GCN_LSTM(
            seq_len=seq_len,
            adj=adj_matrix,
            gc_layer_sizes=[best_trial.params[f"gc_layer_{i}"] for i in range(best_trial.params["num_gc_layers"])],
            gc_activations=[best_trial.params[f"gc_activation_{i}"] for i in range(best_trial.params["num_gc_layers"])],
            lstm_layer_sizes=[best_trial.params[f"lstm_layer_{i}"] for i in range(best_trial.params["num_lstm_layers"])],
            lstm_activations=[best_trial.params[f"lstm_activation_{i}"] for i in range(best_trial.params["num_lstm_layers"])],
            dropout=best_trial.params["dropout"],  # Use the best-tuned dropout value
        )

        x_input, x_output = gcn_lstm.in_out_tensors()
        model = Model(inputs=x_input, outputs=x_output)

        # Define the optimizer again using the best parameters
        if best_trial.params["optimizer"] == "adam":
            optimizer = Adam(
                learning_rate=best_trial.params["lr"],
                beta_1=best_trial.params["adam_beta_1"],
                beta_2=best_trial.params["adam_beta_2"]
            )
        elif best_trial.params["optimizer"] == "sgd":
            optimizer = SGD(
                learning_rate=best_trial.params["lr"],
                momentum=best_trial.params["sgd_momentum"]
            )
        elif best_trial.params["optimizer"] == "rmsprop":
            optimizer = RMSprop(
                learning_rate=best_trial.params["lr"],
                rho=best_trial.params["rmsprop_rho"]
            )

        # Compile model with best parameters
        model.compile(optimizer=optimizer, loss=best_trial.params["loss_function"], metrics=['accuracy'])


        # Now train with the optimal parameters
        final_history = model.fit(
            trainX,
            trainY,
            epochs=epc_number,
            batch_size=best_trial.params["batch_size"],
            shuffle=True,
            verbose=1,
            validation_data=(testX, testY),
        )

        model.summary()

        print(
        "Train loss: ",
        final_history.history["loss"][-1],
        "\nTest loss:",
        final_history.history["val_loss"][-1],
        )

        final_history.history.keys()

        #sg.utils.plot_history(final_history)

        # Plot training & validation loss values
        fig11 = plt.figure(figsize=(6, 6))
        plt.plot(final_history.history['loss'])
        plt.plot(final_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        # Save the plot to a file
        fig11.savefig(f'model_loss_plot_Fold_{ki}.png')  # Change the filename and format as needed
        plt.close(fig11)  # Close the plot to free up memory

        ythat = model.predict(trainX)
        yhat = model.predict(testX)
        yF = model.predict(ForecastX)

        # To rescale back to original
        train_rescref = scaler.inverse_transform(trainY)
        test_rescref = scaler.inverse_transform(testY)
        forecast_rescref = scaler.inverse_transform(yF)
        train_rescpred = scaler.inverse_transform(ythat)
        test_rescpred = scaler.inverse_transform(yhat)

        
        # Save prediction and observation for train and test and forecasting to a text file
        np.savetxt('Forecast.txt', forecast_rescref[:, -6:])

        a_pred = test_rescpred[:, -6:]
        a_true = test_rescref[:, -6:]
        b_pred = train_rescpred[:, -6:]
        b_true = train_rescref[:, -6:]
        np.savetxt(f'a_pred_{ki}.txt', a_pred)
        np.savetxt(f'a_true_{ki}.txt', a_true)
        np.savetxt(f'b_pred_{ki}.txt', b_pred)
        np.savetxt(f'b_true_{ki}.txt', b_true)

        # Predict different scenarios:
        # Directory containing the Excel files
        directory = r'C:\UConn\Ethiopia\GCNLSTM_ERA5_CMIP\SC1'
        # List all files in the directory
        files = os.listdir(directory)
        # Filter Excel files
        excel_files = [file for file in files if file.endswith('.xlsx')]
        # Read each Excel file
        for idx, file in enumerate(excel_files):
            file_path = os.path.join(directory, file)
            df = pd.read_excel(file_path)
            Input_F2 = np.array(df)
            
            F2_scaled = scaler.transform(Input_F2)            

            trainX2, trainY2, testX2, testY2, ForecastX2, ForecastY2 = sequence_data_preparation(
            seq_len, pre_len, train_scaled, test_scaled, F2_scaled.T
            )

            # Forecasting for different month
            yF2 = model.predict(ForecastX2)

            # Rescale:
            forecast_rescref2 = scaler.inverse_transform(yF2)
            #print('forecast_rescref2 shape :', forecast_rescref2.shape)

            file_number = idx  # Add 1 to start numbering from 1
            filename = f'Forecast_{ki}_{file_number}.txt'
            np.savetxt(filename, forecast_rescref2[:, -6:])

            ###in this part we are going to get the results for all stations.
            output_file = f'output_{ki}.txt'
            #output_file = 'output.txt'
            with open(output_file, 'w') as file:
                 for C in range(6):
                     Pred_line = Num_St-6+C

                     # All test result visualization
                     fig3 = plt.figure(figsize=(6, 6))
                     a_pred = test_rescpred[:, Pred_line]
                     a_true = test_rescref[:, Pred_line]
                     plt.scatter(a_true, a_pred, color="b", label="Prediction vs. Observation")
                     # Add the 45-degree center line
                     min_val = min(min(a_true), min(a_pred))
                     max_val = max(max(a_true), max(a_pred))
                     plt.plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--', label='Perfect prediction line')
                     plt.xlabel("Prediction yielding (kg/ha)")
                     plt.ylabel("Observation yielding (kg/ha)")
                     plt.title('Prediction Data vs. Observation Data')
                     plt.legend()
                     fig3.savefig(f'Prediction_scatter_plot_Test_Fold_{ki}_Area_{C}.png')  # Save the plot as a PNG file
                     plt.close(fig3)


                     ##all train result visualization
                     fig4 = plt.figure(figsize=(6, 6))
                     b_pred = train_rescpred[:, Pred_line]
                     b_true = train_rescref[:, Pred_line]
                     plt.scatter(b_true, b_pred, color="b", label="Prediction vs. Observation")
                     # Add the 45-degree center line
                     min_val = min(min(b_true), min(b_pred))
                     max_val = max(max(b_true), max(b_pred))
                     plt.plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--', label='Perfect prediction line')
                     plt.xlabel("Prediction yielding (kg/ha)")
                     plt.ylabel("Observation yielding (kg/ha)")
                     plt.legend()
                     fig4.savefig(f'Prediction_scatter_plot_Train_Fold_{ki}_Area_{C}.png')  # Save the plot as a PNG file
                     plt.close(fig4)

                     # Open the text file for writing
                     # Construct the output file name with the counter value
                     file.write(f"Station: {C}\n")
                     # Calculate and write the variance and mean of the true set
                     true_variance = statistics.variance(a_true)
                     true_mean = statistics.mean(a_true)
                     file.write(f"Variance of true set is: {true_variance}\n")
                     file.write(f"Mean of true set is: {true_mean}\n")

                     # Calculate and write the variance and mean of the pred set
                     a=np.array(a_pred)
                     pred_variance = np.var(a)
                     pred_mean = np.mean(a)
                     file.write(f"Variance of pred set is: {pred_variance}\n")
                     file.write(f"Mean of pred set is: {pred_mean}\n")

                     # Create pandas Series objects for the time series data
                     observational_series = pd.Series(a_true)
                     forecasting_series = pd.Series(a_pred)

                     # Plot the original vs. predicted time series:
                     plt.figure(figsize=(10, 5))  # Set the size of the plot here (width, height)
                     plt.plot(observational_series, label='Observational')
                     plt.plot(forecasting_series, label='Forecasting')
                     plt.xlabel('Station-Year')
                     plt.ylabel('Crop Yielding (kg/ha)')
                     plt.legend()
                     plt.savefig(f'Timeseries_Prediction_Test_Fold_{ki}_Area_{C}.png')
                     plt.close()

                     # Create pandas Series objects for the time series data
                     observational_series = pd.Series(b_true)
                     forecasting_series = pd.Series(b_pred)

                     # Plot the original and moving average time series
                     plt.figure(figsize=(10, 5))  # Set the size of the plot here (width, height)
                     plt.plot(observational_series, label='Observational')
                     plt.plot(forecasting_series, label='Forecasting')
                     plt.xlabel('Station-Year')
                     plt.ylabel('Crop Yielding (kg/ha)')
                     plt.legend()
                     plt.savefig(f'Timeseries_Prediction_Train_Fold_{ki}_Area_{C}.png')
                     plt.close()

                     # Calculate Metrics for TRAIN
                     #Success points
                     def evaluate_forecasting(predictions, observations):
                         predictions = np.array(predictions)  # Ensure predictions is a NumPy array
                         observations = np.array(observations)  # Ensure observations is a NumPy array
                         num_timesteps = len(observations)

                         absolute_errors = []
                         relative_errors = []

                         for i in range(num_timesteps):
                             abs_error = abs(predictions[i] - observations[i])
                             rel_error = abs_error / abs(observations[i]) if observations[i] != 0 else 0  # Set to 0 instead of inf

                             absolute_errors.append(abs_error)
                             relative_errors.append(rel_error)

                         mean_absolute_error = np.mean(absolute_errors)
    
                         # Filter out relative errors that are zero
                         filtered_relative_errors = [re for re in relative_errors if re != 0]
    
                         mean_relative_error = np.mean(filtered_relative_errors) if filtered_relative_errors else 0  # Handle case with no valid errors

                         return mean_absolute_error, mean_relative_error


                     # Absolute bias and Relative bias:
                     AB, RB = evaluate_forecasting(a_pred, a_true)
                     file.write(f"AB Test: {AB}\n")
                     file.write(f"RB Test: {RB}\n")


                     from sklearn.metrics import mean_squared_error
                     MSE = mean_squared_error(a_true, a_pred)
                     RMSE = MSE ** 0.5  # Calculate RMSE

                     # Calculate the mean of the true values
                     mean_true = np.mean(a_true)

                     # To avoid division by zero, check if the mean is greater than zero
                     if mean_true > 0:
                         NRMSE = RMSE / mean_true
                     else:
                         NRMSE = float('inf')  # or set to a specific value, e.g., 0 or None
                     file.write(f"MSE Test: {MSE}\n")
                     file.write(f"RMSE Test: {RMSE}\n")
                     file.write(f"NRMSE Test: {NRMSE}\n")


                     from sklearn.metrics import mean_absolute_error
                     MAE = mean_absolute_error(a_true, a_pred)
                     file.write(f"MAE Test: {MAE}\n")

 
                     corr_matrix = numpy.corrcoef(a_true, a_pred)
                     corr = corr_matrix[0,1]
                     R_sq = corr**2
                     file.write(f"R2 Test: {R_sq}\n")


                     def calculate_nse(y_true, y_pred):
                        # Calculate the mean of the true values
                        y_true_mean = np.mean(y_true)

                        # Calculate the numerator and denominator of the NSE formula
                        numerator = np.sum((y_true - y_pred) ** 2)
                        denominator = np.sum((y_true - y_true_mean) ** 2)

                        # Calculate NSE
                        nse = 1 - (numerator / denominator)

                        return nse
                     nse = calculate_nse(a_true, a_pred)
                     file.write(f"NSE Test: {nse}\n")


                     def calculate_pbias(y_true, y_pred):
                        # Calculate the numerator and denominator of the PBIAS formula
                        numerator = np.sum(y_pred - y_true)
                        denominator = np.sum(y_true)

                        # Calculate PBIAS
                        pbias = (numerator / denominator) * 100

                        return pbias
                     pbias = calculate_pbias(a_true, a_pred)
                     file.write(f"PBIAS Test: {pbias}\n")


                     def calculate_mape(actual, predicted):
                         # Filter out non-zero actual values
                         non_zero_mask = actual != 0
                         actual_non_zero = actual[non_zero_mask]
                         predicted_non_zero = predicted[non_zero_mask]

                         # Calculate MAPE only for non-zero actual values
                         return np.mean(np.abs((actual_non_zero - predicted_non_zero) / actual_non_zero)) * 100

                     mape = calculate_mape(a_true, a_pred)
                     file.write(f"MAPE Test: {mape}\n")


                     def calculate_smape(actual, predicted):
                        numerator = np.abs(actual - predicted)
                        denominator = (np.abs(actual) + np.abs(predicted)) / 2
                        smape = np.mean(numerator / denominator) * 100
                        return smape

                     smape = calculate_smape(a_true, a_pred)
                     file.write(f"SMAPE Test: {smape}\n")


                     # Calculate Metrics for TRAIN
                     #Success points
                     # Calculate Metrics for TRAIN
                     #Success points
                     def evaluate_forecasting(predictions, observations):
                         predictions = np.array(predictions)  # Ensure predictions is a NumPy array
                         observations = np.array(observations)  # Ensure observations is a NumPy array
                         num_timesteps = len(observations)

                         absolute_errors = []
                         relative_errors = []

                         for i in range(num_timesteps):
                             abs_error = abs(predictions[i] - observations[i])
                             rel_error = abs_error / abs(observations[i]) if observations[i] != 0 else 0  # Set to 0 instead of inf

                             absolute_errors.append(abs_error)
                             relative_errors.append(rel_error)

                         mean_absolute_error = np.mean(absolute_errors)
    
                         # Filter out relative errors that are zero
                         filtered_relative_errors = [re for re in relative_errors if re != 0]
    
                         mean_relative_error = np.mean(filtered_relative_errors) if filtered_relative_errors else 0  # Handle case with no valid errors

                         return mean_absolute_error, mean_relative_error


                     # Absolute bias and Relative bias:
                     AB, RB = evaluate_forecasting(b_pred, b_true)
                     file.write(f"AB Train: {AB}\n")
                     file.write(f"RB Train: {RB}\n")

                     #print("FOR TRAIN PERIOD WE HAVE")
                     from sklearn.metrics import mean_squared_error
                     MSE = mean_squared_error(b_true, b_pred)
                     RMSE = MSE ** 0.5  # Calculate RMSE

                     # Calculate the mean of the true values
                     mean_true = np.mean(b_true)

                     # To avoid division by zero, check if the mean is greater than zero
                     if mean_true > 0:
                         NRMSE = RMSE / mean_true
                     else:
                         NRMSE = float('inf')  # or set to a specific value, e.g., 0 or None
                     file.write(f"MSE Train: {MSE}\n")
                     file.write(f"RMSE Train: {RMSE}\n")
                     file.write(f"NRMSE Train: {NRMSE}\n")


                     from sklearn.metrics import mean_absolute_error
                     MAE = mean_absolute_error(b_true, b_pred)
                     file.write(f"MAE Train: {MAE}\n")


                     corr_matrix = numpy.corrcoef(b_true, b_pred)
                     corr = corr_matrix[0,1]
                     R_sq = corr**2
                     file.write(f"R2 Train: {R_sq}\n")


                     def calculate_nse(y_true, y_pred):
                        # Calculate the mean of the true values
                        y_true_mean = np.mean(y_true)

                        # Calculate the numerator and denominator of the NSE formula
                        numerator = np.sum((y_true - y_pred) ** 2)
                        denominator = np.sum((y_true - y_true_mean) ** 2)

                        # Calculate NSE
                        nse = 1 - (numerator / denominator)

                        return nse
                     nse = calculate_nse(b_true, b_pred)
                     file.write(f"NSE Train: {nse}\n")


                     def calculate_pbias(y_true, y_pred):
                        # Calculate the numerator and denominator of the PBIAS formula
                        numerator = np.sum(y_pred - y_true)
                        denominator = np.sum(y_true)

                        # Calculate PBIAS
                        pbias = (numerator / denominator) * 100

                        return pbias
                     pbias = calculate_pbias(b_true, b_pred)
                     file.write(f"PBIAS Train: {pbias}\n")


                     def calculate_mape(actual, predicted):
                         # Filter out non-zero actual values
                         non_zero_mask = actual != 0
                         actual_non_zero = actual[non_zero_mask]
                         predicted_non_zero = predicted[non_zero_mask]

                         # Calculate MAPE only for non-zero actual values
                         return np.mean(np.abs((actual_non_zero - predicted_non_zero) / actual_non_zero)) * 100

                     mape = calculate_mape(b_true, b_pred)
                     file.write(f"MAPE Train: {mape}\n")


                     def calculate_smape(actual, predicted):
                        numerator = np.abs(actual - predicted)
                        denominator = (np.abs(actual) + np.abs(predicted)) / 2
                        smape = np.mean(numerator / denominator) * 100
                        return smape

                     smape = calculate_smape(b_true, b_pred)
                     file.write(f"SMAPE Train: {smape}\n")


        kk = kk+1








  