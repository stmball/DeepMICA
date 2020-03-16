import numpy as np
import itertools as it
import more_itertools as mit
from scipy.stats import ks_2samp
from .ndtest import ks2d2s
import pandas as pd
from .generate import MarkovLog, Network
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score



""" 

    File for ion channel data:
    
    - Comparison of both theoretical models (Network object) diagonalisation and normal form 
    methods, as well as histogram comparison for recorded/simulated data.


"""


def singleDwells(idealisation, nature):
    """

    Calculate the single state dwell times of an idealisation log.

    Args:
        idealisation (list): List of idealisations
        nature (int): State to calculate the dwell times for.

    Returns:
        list: List of state dwell times for given idealisation and state.
    """

    return [len(list(i[1])) for i in filter(lambda x: x[0] == nature, it.groupby(idealisation))]


def dualDwells(idealisation, natures):
    """

    Calculate the combined state dwell times of an idealisation log.

    Args:
        idealisation (list): List of idealisations
        natures (tuple): Ordered tuple of states to find the dwell times for

    Returns:
        list : List of tuples, each containing the dwell times for each corresponding state.
    """
    groupy = map(lambda x: (x[0], len(list(x[1]))), it.groupby(idealisation))
    return [(i[0][1], i[1][1]) for i in filter(lambda x: (x[0][0], x[1][0]) == natures, mit.pairwise(groupy))]


def ks2d2sPrepper(array, natures):
    """

    Helper function for preparing samples for the 2 sample 2 dimensional KS test

    Args:
        array (list): List of idealisations
        natures (tuple): Ordered tuple of states ot find the dwell times for

    Returns:
        tuple: Tuple containg two numpy arrays, corresponding to all x and all y times where (x, y) is natures. 
    """
    dwells = zip(*dualDwells(array, natures))
    return (np.array(i) for i in dwells)


def twoSampleIdealisationTest(idealisation_a, idealisation_b):
    """

    Comparing two different idealisations by comparing the dwell time distributions for open, closed, open-close, close-open.

    Args:
        idealisation_a (list): Idealisation list for model A
        idealisation_b (list): Idealisation list for model B

    Returns:
        dict: Dictionary containing test statistics and p values for each distribution.
    """
    try:
        ks_2samp(singleDwells(idealisation_a, 0), singleDwells(idealisation_b, 0))
        ks_2samp(singleDwells(idealisation_a, 1), singleDwells(idealisation_b, 1))
    except:
        return {
            'Open': [0,0],
            'Closed': [0,0],
            'Open-Closed': [0,0],
            'Closed-Open': [0,0]
        }

    else:
    # TODO: Generalise for any number of states (needs n dim KS test - very hard!)
        return {
            'Open': ks_2samp(singleDwells(idealisation_a, 0), singleDwells(idealisation_b, 0)),
            'Closed': ks_2samp(singleDwells(idealisation_a, 1), singleDwells(idealisation_b, 1)),
            'Open-Closed': ks2d2s(*ks2d2sPrepper(idealisation_a, (0, 1)), *ks2d2sPrepper(idealisation_b, (0, 1)), extra=True),
            'Closed-Open': ks2d2s(*ks2d2sPrepper(idealisation_a, (1, 0)), *ks2d2sPrepper(idealisation_b, (1, 0)), extra=True),
        }


def stateReduce(y_trueArg, y_predictArg, StatesDict):
    """

    Reduce states down into open/closed nature given a common state dictionary

    Returns:
        tuple: Tuple containing state reduction for y_trueArg and y_predictArg

    """
    sortedDict = {k: v for k, v in sorted(StatesDict.items(), key=lambda item: item[0])}
    newTest = []
    newPred = []

    for i in y_trueArg:
        newTest.append(list(sortedDict.values())[i])

    for j in y_predictArg:
        newPred.append(list(sortedDict.values())[j])

    return (newTest, newPred)

def allowPermutations(y_trueArg, y_predictArg, StatesDict, metric):
    mapDict = {v:k for k,v in enumerate(sorted(StatesDict.keys()))}
    
    openIndexes = [mapDict[i] for i in mapDict.keys() if StatesDict[i] == 0]
    closedIndexes = [mapDict[i] for i in mapDict.keys() if StatesDict[i] == 1]


    metrics = []

    for openPerm in it.permutations(openIndexes):
        for closedPerm in it.permutations(closedIndexes):
            mapping = [x for _,x in sorted(zip(openIndexes + closedIndexes, openPerm + closedPerm))]

            newPred = [mapping[i] for i in y_predictArg]
            metrics.append(metric(y_trueArg, newPred))
    return max(metrics)


def MICAStats(y_trueArg, y_predictArg, StatesDict):
    """
        Function for calculating basic statistics for testing model performance, using both reduced and unreduced states.
    """
    
    if isinstance(StatesDict, dict):
        trueIdeal, predictIdeal = stateReduce(y_trueArg, y_predictArg, StatesDict)
        histTest = twoSampleIdealisationTest(trueIdeal, predictIdeal)
        hist_test_stat_list = [i[0] for i in histTest]

        return {
            'Unreduced Kappa': cohen_kappa_score(y_trueArg, y_predictArg),
            'Reduced Kappa': cohen_kappa_score(trueIdeal, predictIdeal),
            'Unreduced F1': f1_score(y_trueArg, y_predictArg, average='micro'),
            'Reduced F1': f1_score(trueIdeal, predictIdeal, average='micro'),
            'histogramMets': histTest,
            'avgHist': np.mean(list(histTest.values()))
        }
    else:
        return {
            'Reduced Kappa': cohen_kappa_score(y_trueArg, y_predictArg),
            'Reduced F1': f1_score(y_trueArg, y_predictArg, average='micro'),
        }

