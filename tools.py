import matplotlib.pyplot as plt
from math import ceil, floor
import numpy as np
from .analysis import stateReduce
from sklearn.preprocessing import MinMaxScaler



def epoch_graph(history, filepath):
    """ 

    Function for plotting and saving a graph of model performance over time.

    Args:
        history (History): History object outputted from model training function.
        filepath (string): File path to save graph to.
    """

    f, ax1 = plt.subplots()
    ax1.plot(history.history['categorical_accuracy'])
    ax1.plot(history.history['val_categorical_accuracy'])
    ax1.autoscale(enable=True, axis='both', tight=True)
    ax1.set_ylim((0, 1))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    f.savefig(filepath)


def comparison_graph(PPData, y_trueArg, y_predictArg, start_time=0, lensec=0.2):
    """
        Function for graphing and comparing model performance with raw data input.
    """

    print('Visualising Data')

    sample_rate = PPData.sampleRate
    lenny = sample_rate * lensec
    start_point = int(sample_rate * start_time)
    end_point = int(start_point + lenny)
    x_data = PPData.scaler.inverse_transform(PPData.x_data.reshape((-1, 1)))[start_point:end_point]
    
    y_true = y_trueArg[start_point:end_point]
    y_predict = y_predictArg[start_point:end_point]

    # Visualisation of completed model, ground truth and raw data
    plt.close()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10))
    ax1.autoscale(enable=True, axis='both', tight=True)
    ax2.autoscale(enable=True, axis='both', tight=True)
    ax3.autoscale(enable=True, axis='both', tight=True)

    # Plot raw data
    ax1.plot(x_data, color='black', label='Raw Data', alpha=0.8)
    ax1.set_xlabel('Time (secs)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_ylim((floor(x_data.min()), ceil(x_data.max())))
    ax1.set_xticks(np.linspace(0, lenny, 11, endpoint=True))
    ax1.set_xticklabels(np.round(np.linspace(
        start_time, start_time + lensec, 11, endpoint=True), ceil(np.log10(lensec)) + 2))

    # Plot state prediction and ground truth
    ax2.plot(y_true, color='blue', label='Ground Truth',
             drawstyle='steps-mid', linestyle='--', alpha=0.8)
    ax2.plot(y_predict, color='red', label='Predicted Values',
             drawstyle='steps-mid', ls=':', alpha=0.8)
    ax2.set_xlabel('Time (secs)')
    ax2.set_ylabel('State')
    ax2.set_ylim((-1, PPData.max_states))
    ax2.set_xticks(np.linspace(0, lenny, 11, endpoint=True))
    ax2.set_xticklabels(np.round(np.linspace(
        start_time, start_time + lensec, 11, endpoint=True), ceil(np.log10(lensec)) + 2))
    ax2.legend()

    # Plot open/closed idealisation
    true_ideal, predict_ideal = stateReduce(
        y_trueArg=y_true, y_predictArg=y_predict, StatesDict=PPData.stateDict)
    ax3.plot(true_ideal, color='blue', label='Ground Truth',
             drawstyle='steps-mid', linestyle='--', alpha=0.8)
    ax3.plot(predict_ideal, color='red', label='Predicted Values',
             drawstyle='steps-mid', linestyle=':', alpha=0.8)
    ax3.set_xlabel('Time (secs)')
    ax3.set_ylabel('Channels Open')
    ax3.set_ylim((-1, 2))
    ax3.set_xticks(np.linspace(0, lenny, 11, endpoint=True))
    ax3.set_xticklabels(np.round(np.linspace(
        start_time, start_time + lensec, 11, endpoint=True), ceil(np.log10(lensec)) + 2))
    ax3.legend()

    return f