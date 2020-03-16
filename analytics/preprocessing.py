import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class PPData:
    """
        Class for returning preprocessed data for CNN models.
    """

    def __init__(self, x_data, y_data, scaler, sampleRate, batch_size, window_size, max_states, stateDict):
        self.x_data = x_data
        self.y_data = y_data
        self.scaler = scaler
        self.sampleRate = sampleRate
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_states = max_states
        self.stateDict = stateDict


def MICAProcess(data, state_column, current_column, max_states, batch_size, window_size, sampleRate, stateDict, tts=False):

    """
        Function that turns data into a minmax scaled, batched dataset ready for input into a deep learning model, with
        options for train test splitting. Also includes "max state consideration" - adds a number of "empty" states to
        one hot encoding to allow for consistent model input size across different state models
    """

    # One hot encoding for state column
    num_states = data[state_column].nunique()
    states, _ = pd.factorize(data[state_column], sort=True)
    data[state_column] = states
    cat_labels = pd.get_dummies(data[state_column], prefix="state_")

    # Max state consideration - add 0 rows to make up for max states
    for missing_state in range(num_states, max_states):
        cat_labels["state" + str(missing_state)] = 0

    # Replace labels column with new cat_labels dataframe
    data.drop([state_column], axis=1, inplace=True)

    if tts:

        # TODO: fix this for tts
        # MUST do train test split here for proper accuracy metrics
        split = int(len(data) * 0.8)
        xtrain, xtest = data[:split], data[split:]
        ytrain, ytest = cat_labels[:split], cat_labels[split:]

        # Truncate data for using with batch size
        trainSize, testSize = len(xtrain)//window_size * \
            window_size, len(xtest)//window_size * window_size
        xtrain, xtest = xtrain[current_column][:
                                               trainSize].values, xtest[current_column][:testSize].values
        ytrain, ytest = ytrain[:trainSize].values, ytest[:testSize].values

        # MinMaxScaling on training data and testing data seperately
        minmaxTrain = MinMaxScaler()
        xtrain = minmaxTrain.fit_transform(xtrain.reshape(-1, 1))

        minmaxTest = MinMaxScaler()
        xtest = minmaxTest.fit_transform(xtest.reshape(-1, 1))

        # Reshaping data for use in batched input
        xtrain = xtrain.reshape((xtrain.shape[0]//window_size, window_size, 1))
        xtest = xtest.reshape((xtest.shape[0]//window_size, window_size, 1))
        ytrain = ytrain.reshape(
            (ytrain.shape[0]//window_size, window_size, max_states))
        ytest = ytest.reshape(
            (ytest.shape[0]//window_size, window_size, max_states))

        return PPData(x_data=(xtrain, xtest), y_data=(ytrain, ytest), scaler=(minmaxTrain, minmaxTest), sampleRate=sampleRate, batch_size=batch_size, window_size=window_size, max_states=max_states, stateDict=stateDict)

    else:


        xdata = data
        ydata = cat_labels
        # Truncate data for batching
        dataSize = len(xdata)//(batch_size * window_size) * (batch_size * window_size) 
        xdata, ydata = xdata[current_column][:dataSize].values, ydata[:dataSize].values
        
        # Minmax scaling
        minmax = MinMaxScaler()
        xdata = minmax.fit_transform(xdata.reshape(-1, 1))

        # Reshaping dat afor use in batched input
        xdata = xdata.reshape((xdata.shape[0]//(batch_size * window_size) , batch_size, window_size, 1))
        ydata = ydata.reshape((ydata.shape[0]//(batch_size * window_size) , batch_size, window_size, max_states))

        return PPData(x_data=xdata, y_data=ydata, scaler=minmax, sampleRate=sampleRate, batch_size=batch_size, window_size=window_size, max_states=max_states, stateDict=stateDict)


def PPFromFile(data_filepath, state_column, current_column, max_states, batch_size, window_size, stateDict, tts=False):
    """
        Wrapper function for performing MICAProcessing on a csv data file.
    """

    # Read file
    data = pd.read_csv(data_filepath)

    # Calculate sample rate
    sampleRate = 1/data['Time'][1]
    return MICAProcess(data, state_column, current_column, max_states, batch_size, window_size, sampleRate, stateDict, tts=tts)


def PPFromMarkovLog(MarkovLog, window_size, max_states, tts=False):

    """
        Wrapper function for performing MICAProcessing on a MarkovLog object
    """

    # Check that continuous history exists
    if not isinstance(MarkovLog.continuous_history, pd.DataFrame):
        return 0
    else:
        return MICAProcess(data=MarkovLog.continuous_history, state_column="State", current_column="Noisy Current", max_states=max_states, window_size=window_size, sampleRate=MarkovLog.sample_rate, stateDict=MarkovLog.state_dict, tts=tts)
