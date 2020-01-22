from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
from math import ceil, floor
import numpy as np


def MICAStats(y_trueArg, y_predictArg, StatesDict):
    """
        Function for calculating basic statistics for testing model performance, using both reduced and unreduced states.
    """
    
   if isinstance(StatesDict, dict):
        things = MICAReduce(y_trueArg, y_predictArg, StatesDict)

        newTest = things['true']
        newPred = things['predict']

        return {
            'Unreduced Kappa': cohen_kappa_score(y_trueArg, y_predictArg),
            'Reduced Kappa': cohen_kappa_score(newTest, newPred),
            'Unreduced F1': f1_score(y_trueArg, y_predictArg, average='micro'),
            'Reduced F1': f1_score(newTest, newPred, average='micro')
        }
    else:
        return {
            'Reduced Kappa': cohen_kappa_score(y_trueArg, y_predictArg),
            'Reduced F1': f1_score(y_trueArg, y_predictArg, average='micro'),
        }
    
def MICAGraph(PPData, y_trueArg, y_predictArg, LENNY=500, lensec = 0.2, ):
    """
        Function for graphing and comparing model performance with raw data input.
    """
    print('Visualising Data')
    # Visualisation of completed model, ground truth and raw data
    plt.close()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10))
    ax1.autoscale(enable=True, axis='both', tight=True)
    ax2.autoscale(enable=True, axis='both', tight=True)
    ax3.autoscale(enable=True, axis='both', tight=True)
    
    # Plot raw data
    ax1.plot(PPData.scaler.inverse_transform(PPData.x_data.reshape((-1, 1)))
             [:LENNY], color='black', label='Raw Data', alpha=0.8)
    ax1.set_xlabel('Time (secs)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_ylim((floor(PPData.scaler.inverse_transform(PPData.x_data.reshape(
        (-1, 1)))[:LENNY].min()), ceil(PPData.scaler.inverse_transform(PPData.x_data.reshape((-1, 1)))[:LENNY].max())))
    ax1.set_xticks(np.linspace(0, LENNY, 10, endpoint=False))
    ax1.set_xticklabels(np.round(np.linspace(
        0, lensec, 10, endpoint=False), ceil(np.log10(lensec)) + 2))
    # Plot ground truth
    ax2.plot(y_trueArg[:LENNY], color='blue',
             label='Ground Truth', drawstyle='steps-mid', linestyle='--')
    ax2.set_xlabel('Time (secs)')
    ax2.set_ylabel('State')
    ax2.set_ylim((-1, len(PPData.max_states)))
    ax2.set_xticks(np.linspace(0, LENNY, 10, endpoint=False))
    ax2.set_xticklabels(np.round(np.linspace(
        0, lensec, 10, endpoint=False), ceil(np.log10(lensec)) + 2))
    # Plot predicted values
    ax2.plot(y_predictArg[:LENNY], color='red',
             label='Predicted Values', drawstyle='steps-mid', ls=':')
    ax2.set_xlabel('Time (secs)')
    ax2.set_ylabel('State')
    ax2.legend()
    # Plot open/closed
    things = MICAReduce(
        y_trueArg=y_trueArg, y_predictArg=y_predictArg, StatesDict=stateDict)
    ax3.plot(things['true'][:LENNY], color='blue',
             label='Ground Truth', drawstyle='steps-mid', linestyle='--')
    ax3.plot(things['predict'][:LENNY], color='red',
             label='Predicted Values', drawstyle='steps-mid', linestyle=':')
    ax3.set_xlabel('Time (secs)')
    ax3.set_ylabel('Channels Open')
    ax3.set_ylim((-1, 2))
    ax3.set_xticks(np.linspace(0, LENNY, 10, endpoint=False))
    ax3.set_xticklabels(np.round(np.linspace(
        0, lensec, 10, endpoint=False), ceil(np.log10(lensec)) + 2))
    ax3.legend()
    return f
