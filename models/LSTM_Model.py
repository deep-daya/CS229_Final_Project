"""
Recurrent neural network model for electricity load
"""
import os
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics


class LSTM_Model:

    def __init__(self, df=None, window=24, layers=1, hidden_inputs=50, last_layer="Dense", scaler="Standard", epochs=30,
                 activation="tanh"):
        self.window = window
        self.layers = layers
        self.hidden_inputs = hidden_inputs
        self.last_layer = last_layer
        self.scaler = scaler
        self.df = df
        self.epochs = epochs
        self.activation = activation
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.X_compare = None
        self.model = None
        self.path = None
        self.date_array = None
        self.train_errors = []
        self.val_errors = []
        self.test_errors = []
        self.outputpath = None
        self.train_samples = []

    def scale_columns(self, col_name_list):
        """Scale columns named in col_name_list to mean zero, sd one,
        and return the individual scaler objects in a dictionary.
        Args:
            df: pandas DataFrame with columns to be scaled
            col_name_list: list of string column names denoting columns
                to be scaled
        Returns: Tuple of two objects. The first is the scaled dataframe.
        The second is a dictionary of scaler objects, the keys are the column names
        of the column scaled by each scaler.
        """
        scaler_dict = {}
        for cur_name in col_name_list:
            if self.scaler == "Standard":
                cur_scaler = StandardScaler()
            else:
                cur_scaler = MinMaxScaler()
            cur_data = self.df.loc[:, cur_name].values
            cur_data = np.expand_dims(cur_data, axis=1)
            cur_scaler.fit(cur_data)
            self.df.loc[:, cur_name] = cur_scaler.transform(cur_data).squeeze()
            scaler_dict[cur_name] = cur_scaler
        return (scaler_dict)

    def create_window_data(self, data, keep_whole_window=True):
        """Preprocess data to match RNN model specification

        Args:
            data: matrix of inputs for the RNN (num_steps, num_features)
            window: the width of the look-back window used for the RNN
            keep_whole_window: flag controlling whether the whole window
                will be preserved in each entry of the output, or only
                the current element at the end of the window.
        Returns: 3-dimensional tensor (np.ndarray) of dimensions
        (num_steps, window [or 1 if keep_whole_window is False], num_features)
        ready for input to an LSTM.
        """
        num_steps = data.shape[0] - self.window + 1
        num_features = data.shape[1]
        if keep_whole_window is False:
            keep_nelems = 1
        else:
            keep_nelems = self.window
        arr = np.zeros((num_steps, keep_nelems, num_features))
        for i in range(self.window, data.shape[0]+1):
            cur_window = data[(i - keep_nelems):i, :]
            arr[i - self.window, :, :] = cur_window
        arr = arr.squeeze()
        return arr

    def define_fit_vanilla_lstm(self):
        """Defines a Vanilla LSTM model using Keras
        Args:
            window: width of the look-back window used for the model's
                input data
            num_features: number of features in the input data
            units: dimension of "memory" space
        Returns: keras Sequential model object
        """
        model = keras.models.Sequential()
        num_features = len(self.X_train.columns) - 1
        if self.layers == 1:
            # TO DO ADD ACTIVATION
            model.add(keras.layers.LSTM(self.hidden_inputs, activation=self.activation, return_sequences=False,
                                        input_shape=(self.window, num_features)))
        else:
            for i in range(self.layers):
                model.add(keras.layers.LSTM(self.hidden_inputs, activation=self.activation, return_sequences=True,
                                            input_shape=(self.window, num_features)))
            model.add(keras.layers.LSTM(self.hidden_inputs , activation=self.activation))
        if self.last_layer == "Dense":
            model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train_test_split(self):
        # Create TimeSeriesSplitter
        tscv = TimeSeriesSplit(n_splits=5)
        X = self.df

        # Find X_train and X_compare
        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            X_compare_2020 = X.iloc[test_index]

        # Find X_val+test
        X_val = X_train.loc[(X_train.index.date > datetime.date(2019, 9, 30))]

        # Find X_Train
        X_train = X_train.loc[~X_train.index.isin(X_val.index)]

        # Find exact X_val and X_test
        tscv = TimeSeriesSplit(n_splits=2)
        for train_index, test_index in tscv.split(X_val):
            X_val_2 = X_val.iloc[train_index]
            X_test_2_month = X_val.iloc[test_index]

        self.X_train = X_train
        self.X_test = X_test_2_month
        self.X_val = X_val_2
        self.X_compare = X_compare_2020

    def run_experiment(self, region, city, path, test_on_split=False, folds = 6,
                       input_keep_cols=['hour', 'weekday', 'weekend', 'pre_weekend',
                                        'post_weekend', 'holiday', 'dwpc', 'relh', 'sped', 'tmpc', 'load','week_of_year','sin_week_of_year','cos_week_of_year','sin_hour','cos_hour']):
        # Read appropriate CSV into a dataframe
        filepath = os.path.join(path, f'{region}_{city}_third_pass.csv')
        df = pd.read_csv(filepath)
        self.path = path
        self.date_array = [i.strftime('%Y-%m-%d %H:%M:%S') for i in pd.to_datetime(df.date_hour)]
        print("Made DataFrame")
        # Create datetime index
        df.index = pd.to_datetime(df.date_hour)
        print(df.columns)
        df_select = df.loc[:, input_keep_cols]

        # Set df to extracted dataset
        self.df = df_select

        # Scale the data
        float_cols = list(df_select.columns[df_select.dtypes == np.float64])
        scaler_dict = self.scale_columns(float_cols)

        print("Splitting and Training")
        # Split train and test data
        if test_on_split:
            self.test_on_splits(scaler_dict, folds)
        else:
            self.train_test_split()
            self.fit_model_and_predict(scaler_dict)

    def fit_model_and_predict(self, scaler_dict, train_too=False):
        # Split data into input and target variables and create tensors
        # For Train data
        date_array = [i.strftime('%Y-%m-%d %H:%M:%S') for i in pd.to_datetime(self.X_val.iloc[self.window-1:].index.values)]
        X_train = self.X_train
        y_train = X_train["load"].to_numpy()
        y_train = np.expand_dims(y_train, axis=1)
        X_train = X_train.drop("load", axis=1).to_numpy()
        X_train_tensor = self.create_window_data(X_train)
        y_train_tensor = self.create_window_data(y_train, keep_whole_window=False)
        self.train_samples.append(len(X_train_tensor))
        # For test Data
        X_test = self.X_test
        y_test = X_test['load'].to_numpy()
        y_test = np.expand_dims(y_test, axis=1)
        X_test = X_test.drop('load', axis=1).to_numpy()
        X_test_tensor = self.create_window_data(X_test)
        y_test_tensor = self.create_window_data(y_test, keep_whole_window=False)

        # For Validation Data
        X_val = self.X_val
        y_val = X_val['load'].to_numpy()
        y_val = np.expand_dims(y_val, axis=1)
        X_val = X_val.drop('load', axis=1).to_numpy()
        X_val_tensor = self.create_window_data(X_val)
        y_val_tensor = self.create_window_data(y_val, keep_whole_window=False)
        if train_too:
            assert round(100 * len(y_val_tensor)/ (len(y_val_tensor) + len(y_train_tensor))) == 20
        # For Compare Data
        X_compare = self.X_compare
        y_compare = X_compare['load'].to_numpy()
        y_compare = np.expand_dims(y_compare, axis=1)
        X_compare = X_compare.drop("load", axis=1).to_numpy()
        X_compare_tensor = self.create_window_data(X_compare)
        y_compare_tensor = self.create_window_data(y_compare, keep_whole_window=False)

        print("Defining and Training Model")
        # Define Model
        self.define_fit_vanilla_lstm()

        # Train Model
        self.model.fit(X_train_tensor, y_train_tensor, epochs=self.epochs)

        # Make Prediction
        y_pred = self.model.predict(X_val_tensor)
        y_val_unscaled = scaler_dict['load'].inverse_transform(y_val_tensor)
        y_pred_unscaled = scaler_dict['load'].inverse_transform(y_pred)
        if train_too:
            # Find unscaled data and calculate metrics
            date_array = [i.strftime('%Y-%m-%d %H:%M:%S') for i in pd.to_datetime(self.X_val.iloc[self.window-1:].index.values)]
            date_array_train = [i.strftime('%Y-%m-%d %H:%M:%S') for i in pd.to_datetime(self.X_train.iloc[self.window-1:].index.values)]

            y_pred_train = self.model.predict(X_train_tensor)
            y_train_unscaled = scaler_dict['load'].inverse_transform(y_train_tensor)
            y_pred_train_unscaled = scaler_dict['load'].inverse_transform(y_pred_train)
            self.calc_metrics(y_val_unscaled, y_pred_unscaled, date_array, scaler_dict,save_outputs=True,
                              train_y_true=y_train_unscaled,
                              train_y_pred=y_pred_train_unscaled, train_dates_arr=date_array_train, swoth = True)
        else:
            self.calc_metrics(y_val_unscaled, y_pred_unscaled, date_array, scaler_dict,save_outputs=True)

    #         #Find unscaled data and calculate metrics
    #         y_pred = self.model.predict(X_test_tensor)
    #         y_test_unscaled = scaler_dict['load'].inverse_transform(y_test_tensor)
    #         y_pred_unscaled = scaler_dict['load'].inverse_transform(y_pred)
    #         self.calc_metrics(y_pred_unscaled, y_test_unscaled)

    #         #Find unscaled data and calculate metrics
    #         y_pred = self.model.predict(X_compare_tensor)
    #         y_compare_unscaled = scaler_dict['load'].inverse_transform(y_compare_tensor)
    #         y_pred_unscaled = scaler_dict['load'].inverse_transform(y_pred)

    #         self.calc_metrics(y_pred_unscaled, y_compare_unscaled)

    def hyperparameter_dict(self, output_dir):
        hyperparams = {"window": self.window, "layers": self.layers, "hidden_inputs": self.hidden_inputs,
                       "epochs": self.epochs}
        filename = 'hyperparameters.json'
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(hyperparams, f, indent=4)

    def test_on_splits(self, scaler_dict, folds = 6):
        tscv = TimeSeriesSplit(n_splits=5)
        X = self.df

        # Find X_train and X_compare
        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            X_compare_2020 = X.iloc[test_index]

        # Find X_test
        X_test = X_train.loc[(X_train.index.date > datetime.date(2019, 12, 31))]

        # Find X_val+train
        X_train_val = X_train.loc[~X_train.index.isin(X_test.index)]
        train_sample_size = []
        # Find exact X_val and X_test
        train_ind, test_ind = self.split_for_splits(folds, X_train_val)
        for i in range(len(train_ind)):
            X_train_2 = X_train_val.iloc[:train_ind[i]]
            X_val_2 = X_train_val.iloc[train_ind[i]: train_ind[i] + test_ind[i]]
            self.X_train = X_train_2
            train_sample_size.append(len(X_train_2))
            self.X_test = X_test
            self.X_val = X_val_2
            self.X_compare = X_compare_2020
            self.fit_model_and_predict(scaler_dict, train_too=True)

        filename_l = ["MSE","RMSE","MAPE","MSE_Scaled","RMSE_Scaled","MAPE_Scaled"]
        for i in range(len(filename_l)):
            train_error = [self.train_errors[j][i] for j in range(len(self.train_errors))]
            val_error = [self.val_errors[j][i] for j in range(len(self.val_errors))]
            plt.figure()
            plt.plot(self.train_samples, train_error,label="training error")
            plt.plot(self.train_samples, val_error, label="validation error")
            plt.legend()
            plt.xlabel("Training Size (Number of Windows)")
            plt.ylabel("Error")
            plt.title("{0} vs Training Size".format(filename_l))
            filename = 'error_scatter_test_splits_{0}.png'.format(filename_l[i])
            plt.savefig(os.path.join(self.outputpath, filename))
            plt.close()


    def split_for_splits(self, folds, data):
        nsplits_len_train = [i * (len(data)-23) // (folds + 1) + (len(data)-23) % (folds + 1) - 1 if i* (len(data)-23) // (folds + 1) <= int(0.8 * (len(data)-23) // 1) else int(0.8*(len(data)-23)) for i in np.arange(1, folds+1)]
        nsplits_len_test = [int((i -23) / 0.8 * 0.2 // 1) + 23 for i in nsplits_len_train]
        return nsplits_len_train, nsplits_len_test

    def calc_metrics(self, y_true, y_pred, dates_arr, scaler_dict,save_outputs=True,
                     train_y_true=None, train_y_pred=None, train_dates_arr=None, swoth = False):
        '''
        Calculates some metrics and plots for current run of an LSTM and saves them
        to a folder under the root/models s ubdirectory
        Parameters
        ----------
        y_true : numpy array of shape (m,1)
            True true load values
        y_pred : numpy array of shape (m,1)
            Predicted load values
        dates_arr : numpy array of shape (m,1)
            Each entry in numpy array should contain a unique date_hour string in
            the format: 'YYYY-MM-DD HH:MM:SS'
        hyperparams : dictionary
            Contains hyperparameter specifications
        save_outputs : bool, optional
            If True, will save outputs to a generated timestamped directory under
            the root/models subdirectory
        Returns
        -------
        None.
        '''

        output_dir = ''  # Placeholder if save_outputs is False
        if save_outputs:
            path = os.path.join(os.getcwd(), "model")
            if not os.path.exists(path):
                os.mkdir(path)
                print("Creating model directory")
            else:
                print("Directory exists, adding to directory")
            # Path to data directory contaning CSVs
            output_dir = path

            run_datetime = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
            new_dir_name = f'run_{run_datetime}'

            if swoth:
                output_dir = os.path.join(output_dir, 'split_train_val')
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                self.outputpath = output_dir
            else:
                pass
            # Current date/time
            output_dir = os.path.join(output_dir, new_dir_name)
            os.mkdir(output_dir)

        # Error statistics #######################################################
        # self.outputpath = output_dir
        self.raw_metrics(y_true, y_pred, output_dir, scaler_dict, save_outputs=save_outputs, prefix='val', swoth = swoth)
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.raw_metrics(train_y_true, train_y_pred, output_dir, scaler_dict ,save_outputs=save_outputs, prefix='train',swoth = swoth)

        # Scatter plot of Predicted vs True Load #################################

        self.plot_scatter(y_true, y_pred, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.plot_scatter(train_y_true, train_y_pred, output_dir, save_outputs=save_outputs, prefix='train')

        # Timeseries Plot (automatically only plots first two weeks of data) ######

        self.plot_timeseries(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.plot_timeseries(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs,
                            prefix='train')

        # Save current run's hyperparameters to JSON file ########################
        if save_outputs:
            self.hyperparameter_dict(output_dir)
        # Save arrays of predicted and true load value to CSV ####################
        self.save_CSV(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.save_CSV(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs, prefix='train')

        return

    def raw_metrics(self, y_true, y_pred, output_dir, scaler_dict,save_outputs=True, prefix='val', swoth = False):
        '''
        Various useful regression metrics, the following link is a helpful reference
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        '''

        print('------------------------------------------------------------------')
        print(f'{prefix} raw metrics')
        print('------------------------------------------------------------------')

        #UnScaled
        MSE = metrics.mean_squared_error(y_true, y_pred)
        RMSE = metrics.mean_squared_error(y_true, y_pred, squared=False)

        Mean_abs_error = metrics.mean_absolute_error(y_true, y_pred)
        Median_abs_error = metrics.median_absolute_error(y_true, y_pred)

        Max_error = metrics.max_error(y_true, y_pred)
        R2_score = metrics.r2_score(y_true, y_pred)

        y_true_mape = y_true.reshape(len(y_true), 1)
        MAPE = (sum(abs(y_true_mape - y_pred) / y_true_mape) / len(y_true_mape))[0]

        #Scaled
        y_true_scaled = scaler_dict["load"].transform(y_true.reshape(-1, 1))
        y_pred_scaled = scaler_dict["load"].transform(y_pred.reshape(-1, 1))
        MSE_scaled = metrics.mean_squared_error(y_true_scaled, y_pred_scaled)
        RMSE_scaled = metrics.mean_squared_error(y_true_scaled, y_pred_scaled, squared=False)


        y_true_scaled = y_true.reshape(len(y_true_scaled),1)
        MAPE_scaled = (sum(abs(y_true_scaled - y_pred_scaled)/y_true_scaled)/len(y_true_scaled))[0]

        metrics_str = ''
        metrics_str += f'Mean Squared Error: {MSE:.2f}\n'
        metrics_str += f'Root Mean Squared Error: {RMSE:.2f}\n'
        metrics_str += f'Mean Absolute Error: {Mean_abs_error:.2f}\n'
        metrics_str += f'Median Absolute Error: {Median_abs_error:.2f}\n'
        metrics_str += f'Max Error: {Max_error:.2f}\n'
        metrics_str += f'R2 Score: {R2_score:.3f}\n'
        metrics_str += f'Mean Absolute Percentage Error (MAPE): {MAPE:.4f}\n'
        metrics_str += f'Mean Squared Error Scaled: {MSE_scaled:.2f}\n'
        metrics_str += f'Root Mean Squared Error Scaled: {RMSE_scaled:.2f}\n'
        metrics_str += f'Mean Absolute Percentage Error Scaled: {MAPE_scaled:.4f}\n'

        print(metrics_str)

        if save_outputs:
            filename = f'{prefix}_error_stats.txt'
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(metrics_str)

        if swoth:
            if prefix == "val":
                self.val_errors.append([MSE, RMSE, MAPE, MSE_scaled, RMSE_scaled, MAPE_scaled])
            elif prefix == "train":
                self.train_errors.append([MSE, RMSE, MAPE, MSE_scaled, RMSE_scaled, MAPE_scaled])
            elif prefix == "test":
                self.test_errors.append([MSE, RMSE, MAPE, MSE_scaled, RMSE_scaled, MAPE_scaled])

        return

    def plot_scatter(self, y_true, y_pred, output_dir, save_outputs=True, prefix='val'):
        '''
        Scatter plot of predicted vs true load, ideal is a line of slope 1
        '''

        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Load (MW)')
        plt.ylabel('Predicted Load (MW)')
        plt.title('Predicted vs True Load')

        if save_outputs:
            filename = f'{prefix}_scatter.png'
            plt.savefig(os.path.join(output_dir, filename))

        return

    def plot_timeseries(self, y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
        '''
        Timeseries plot of predicted/true load vs datetime to see how well the
        prediction values match the true curve
        '''

        dates_arr_plt = self.to_dt(dates_arr)

        plt.figure()

        start_ind = 0
        end_ind = 336  # First two weeks
        end_ind = min(end_ind, len(dates_arr_plt))

        plt.plot_date(dates_arr_plt[start_ind:end_ind], y_true[start_ind:end_ind], 'b:', label='True')
        plt.plot_date(dates_arr_plt[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Load (MW)')
        plt.title('2-Week Predicted vs True Load')
        plt.legend()

        if save_outputs:
            filename = f'{prefix}_timeseries.png'
            plt.savefig(os.path.join(output_dir, filename))

        return

    def save_CSV(self, y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
        '''
        Save CSV file of predicted and true load
        '''

        if save_outputs:
            dates_arr = np.reshape(dates_arr, (len(dates_arr), 1))
            y_true = y_true.reshape(len(y_true),1)
            y_pred = y_pred.reshape(len(y_pred),1)
            df_out = np.hstack([dates_arr, y_true, y_pred])

            header = ['date_hour', 'True Load', 'Predicted Load']
            filename = f'{prefix}_pred_array.csv'
            output_filepath = os.path.join(output_dir, filename)
            pd.DataFrame(df_out).to_csv(output_filepath, header=header, index=False)

        return

    def get_datetime(self, dt_str):
        '''
        Helper function to get a datetime object from a date_hour string
        '''

        dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

        return dt_obj

    def to_dt(self, dt_arr):
        '''
        Helper function to convert array of date_hour timestrings to datetime objects
        '''

        # If already a list of datetime objects, just return the list
        if isinstance(dt_arr[0], datetime.datetime):
            return dt_arr

        # Other wise convert from strings to datetime objects
        N = len(dt_arr)

        return_arr = [''] * N

        j = 0
        for i in range(N):
            return_arr[j] = self.get_datetime(dt_arr[i])
            j += 1

        return return_arr


def main():
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'

    # Train model and predict
    lstm = LSTM_Model(df=None, window=24, layers=2, hidden_inputs=50, last_layer="Dense", scaler="Standard", epochs=20,
                      activation="relu")
    lstm.run_experiment(region, city, os.getcwd(),test_on_split=True,folds=7)

    return


if __name__ == "__main__":
    main()