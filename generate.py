import numpy as np
import networkx as nx
import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import itertools as it
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from math import ceil
from collections import defaultdict
from functools import reduce
import networkx as nx
from colorednoise import powerlaw_psd_gaussian

from .fwd_bkw import viterbi, forward_backward, generate_emission_matrix, normalise_trans_matrix



"""

    Generation and simulation of markovian ion channel data models,
    plotting diagrams of models, simulated data, and open/closed dwell time distributions

"""


def _get_markov_edges(Q):
    """

    Helper function for getting the Markov edges from a Pandas Dataframe converted Network

    Args:
        Q (pandas.DataFrame): Markov transition matrix in pandas dataframe form

    Returns:
        dict: Dictionary of edges

    """

    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


def directSum(a, b):
    """

    Calculating the direct sum of two matricies

    Args:
        a (numpy.darray): First matrix in direct sum
        b (numpy.darray): Second matrix in direct sum

    Returns:
        numpy.darray: Direct sum result

    """

    dsum = np.zeros(np.add(a.shape, b.shape))
    dsum[:a.shape[0], :a.shape[1]] = a
    dsum[a.shape[0]:, a.shape[1]:] = b
    return dsum


def getDiagonaliser(matrix):
    """

    Helper function for finding diagonalisation matrix that has row sums equalling 1

    Args:
        matrix (numpy.darray): Matrix for finding diagonaliser

    Returns:
        numpy.darray: Diagonalisation matrix

    """

    _, v = np.linalg.eig(matrix)
    normalizer = np.linalg.inv(v) @ np.ones((v.shape[0]))
    normalizer = np.tile(normalizer, (matrix.shape[0], 1))
    preout = v * normalizer
    a = list(range(preout.shape[0]))
    b = sorted(list(range(preout.shape[0])),
               key=lambda item: preout[item, item])
    out = np.zeros_like(preout)
    for i, j in zip(a, b):
        out[:, j] = preout[:, i]
    print(out)
    return out


def zeroSmalls(matrix):
    """

    Helper function for zeroing values smaller than 1e-8 in a matrix, used for floating point artifacts

    Args:
        matrix (numpy.darray): Matrix for zero-ing

    Returns:
        numpy.darray: Input matrix but with small elements zero-d

    """

    low_value_flag = abs(matrix) < 1e-8
    newMatrix = matrix.copy()
    newMatrix[low_value_flag] = 0
    return newMatrix


def possibleCycle(network, openStates, closedStates):
    """

    Helper function for finding possible edges to create cycles in a network,
    assuming that we can only connect different natures together (not open-open
    for example)

    Args:
        network (Network): Network we are trying to find cycles for
        openStates (list): List of open states in the network
        closedStates (list): List of closed states in the network

    Returns:
        tuple: Tuple containing whether a cycle is possible, and edges that generate
        an allowed cycle

    """

    flag = False

    candidateCycles = []
    for i, j in it.product(openStates, closedStates):
        if network[i][j] == 0:
            flag = True
            candidateCycles.append((i, j))

    return flag, candidateCycles


def createCycle(network, stateClasses):
    """ 

    Helper function for creating new cycles

    Args:
        network (Network): Network object we are adding a cycle to
        stateClasses (list): List of tuples describing candidate edges to add

    Returns:
        Network: New network with cycle added   

    """

    i, j = random.choice(stateClasses)
    network[i][j] = 1
    network[j][i] = 1
    return network


def checkRandomCanonical(randoms, opens, closes):
    for block in [randoms[0:opens, opens:], randoms[opens:, 0:opens]]:
        if not np.array_equal(np.sum(block, axis=0), np.sort(np.sum(block, axis=0))):
            return False
        elif not np.array_equal(np.sum(block, axis=1), np.sort(np.sum(block, axis=1))):
            return False
        else:
            return True


def check_markov_form(trans_matrix):
    for idx, row in enumerate(trans_matrix):
        if row[idx] != -sum(np.delete(row, idx)):
            return False
    else:
        return True


def sample_from_rate(rate):
    """

    Helper function for sampling from the exponential distribution with a given rate

    Args:
        rate (float): Exponential distribution rate parameter

    Returns:
        float: Random sample from the exponential distribution

    """

    # * Note that numpy uses 1/rate exp(- 1/rate x) as the distribution because it's weird

    if rate <= 0:
        return np.inf
    else:
        return np.random.exponential(scale=1/rate)


class SquareRootScale(mscale.ScaleBase):
    """

    ScaleBase class for generating square root scale.

    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis=axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


class Network:

    def __init__(self, states, **kwargs):
        """ 

        Network object initialisation - can accept a number of arguements for 
        predefining a network, or leaving it empty for later population.

        Args:
            states (int): Number of states of the network

        Kwargs:
            adjMatrix (np.darray): Predefined adjacency matrix for network
            trans_matrix (np.darray): Predefined transition rate matrix for network

        Raises:
            TypeError: Adjacency matrix must be numpy array - error raised if 
            adjacency matrix kwarg is not the right type.

            TypeError: Adjacency matrix dimensions must agree with number of states
            - raised if the matrix is not a square matrix with the same number of
            rows and columns as the state arguement.

            TypeError: Adjacency matrix must be symmetric - error raised if adjacency
            matrix is not symmetric. We assume that graphs are undirected

            TypeError: Adjacency matrix must only have one and zero entries - raised
            if adjacency matrix has entries other than one and zero. We assume that graphs
            are simple.

            TypeError: Transition matrix must be numpy array - error raised if 
            transition matrix kwarg is not the right type.

            TypeError: Transition matrix dimensions must agree with number of states -
            error raised if the matrix is not square with the same number of rows and columns
            as the state arguement.

        Returns:
            Network: Network object describing the defined network. 

        """

        self.states = states

        # Checks to see if the adjacency and transition rate matricies are allowed.
        if 'adjMatrix' in kwargs:
            adjMatrix = kwargs['adjMatrix']
            # Check type is numpy array
            if not isinstance(adjMatrix, np.ndarray):
                raise TypeError('Adjacency matrix must be numpy array.')

            # Check adjMatrix is square with rows and columns equal to the state arguement
            elif adjMatrix.shape != (states, states):
                raise TypeError(
                    'Adjacency matrix dimensions must agree with number of states.')

            # Check for symmetry
            elif np.array_equal(adjMatrix, adjMatrix.T):
                raise TypeError('Adjacency matrix must be symmetric!')

            # Check that matrix has only 0s and 1s
            elif np.any(adjMatrix != 0 and adjMatrix != 1):
                raise TypeError(
                    'Adjacency matrix must only have one and zero entries!')

            # If all tests are clear, set adjMatrix attribute to the kwarg
            else:
                self.adjMatrix = adjMatrix

        # If no adjacency matrix is given, initalise an empty zero adjacency matrix
        else:
            self.adjMatrix = np.zeros((states, states))

        if 'trans_matrix' in kwargs:
            trans_matrix = kwargs['trans_matrix']
            # Check type is numpy array
            if not isinstance(trans_matrix, np.ndarray):
                raise TypeError('Transition matrix must be numpy array.')

            # Check trans_matrix is square with rows and columns equal to the state arguement
            elif trans_matrix.shape != (states, states):
                raise TypeError(
                    'Transition matrix dimensions must agree with number of states.')

            elif not check_markov_form(kwargs['trans_matrix']):
                raise ValueError('Transition matrix must be in proper Markov form - entries on the diagonal should equal the negative sum of the other row elements.')
            # If all tests are clear, set trans_matrix attribute to the kwarg
            else:
                self.trans_matrix = trans_matrix
                if 'adjMatrix' in kwargs and self.adjMatrix != self.generateAdjMatrix():
                    raise Warning(
                        'Transition matrix has non zero rates in positions where adjacency matrix has ones.')
                if 'adjMatrix' not in kwargs:
                    self.adjMatrix = self.generateAdjMatrix()

        # If no transition matrix is given, initalise an empty zero transition matrix
        else:
            self.trans_matrix = np.zeros((states, states))

        if 'state_dict' in kwargs:
            state_dict = kwargs['state_dict']
            # Check type is numpy array
            if not isinstance(state_dict, dict):
                raise TypeError('State dictionary must be Python dictionary')

            # If all tests are clear, set state_dict attribute to the kwarg
            else:
                self.state_dict = state_dict

        # If no transition matrix is given, initalise an empty zero transition matrix
        else:
            self.state_dict = {}

    def check_markov_form(self):
        return check_markov_form(self.trans_matrix)

    def randomiseAdj(self):
        """ 

        Method for randomising the Adjacency matrix, within the contraint that no cycles are created

        """

        # Start with no nodes connected, all nodes disconnected
        connectedNodes = []
        disconnectedNodes = list(range(self.states))

        # Select initial node and add it to connected nodes list
        selected = np.random.choice(disconnectedNodes)
        connectedNodes.append(selected)
        disconnectedNodes.remove(selected)

        # Iterate through list of connected and disconnected nodes, randomly connecting two together
        for _ in range(self.states - 1):
            joiner = np.random.choice(connectedNodes)
            joinee = np.random.choice(disconnectedNodes)
            self.adjMatrix[joinee, joiner] = 1
            self.adjMatrix[joiner, joinee] = 1
            connectedNodes.append(joinee)
            disconnectedNodes.remove(joinee)

    def randomiseWeights(self, mag, preserveCanonical=False):
        """

        Method for randomising the transition rate matrix for a network up to a given magnitude

        Args:
            mag (float): Ceiling value for random transition rate entries

        """

        adj = self.adjMatrix
        # Generate random numpy matrix with maximal possible value of the mag arguement

        if preserveCanonical:
            opens = list(self.state_dict.values()).count(0)
            closes = list(self.state_dict.values()).count(1)
            flag = True
            while flag:
                # TODO: Extend to multi channel
                randoms = np.multiply(adj, np.random.rand(*adj.shape) * mag)
                flag = not checkRandomCanonical(randoms, opens, closes)
        else:
            randoms = np.multiply(adj, np.random.rand(*adj.shape) * mag)
        # Fix the diagonal to be the negative sum of all other entries in that row
        for i in range(randoms.shape[0]):
            randoms[i, i] = -1 * (np.sum(randoms[i, :]) - randoms[i, i])

        # Set new transition matrix
        self.trans_matrix = randoms
        return self

    def randomiseStates(self):
        """

        Method for randomising the state natures of a network under the constraint that no two
        connected nodes can be of the same nature.

        """

        # TODO: Add generalisation for any number of channels

        # Initialising empty sets for the open and closed states
        openStates = set({})
        closedStates = set({})

        # Randomly choose a stating state to iterate from
        startState = np.random.randint(0, self.states)

        # Define the above state as open
        openStates.add(startState)

        # Iterate through all states in the network, alternating between adding a closed or open state
        while len(openStates) + len(closedStates) < self.states:
            for node in openStates:
                closedStates = closedStates.union(
                    set(np.where(self.adjMatrix[node] == 1)[0]))
            for node in closedStates:
                openStates = openStates.union(
                    set(np.where(self.adjMatrix[node] == 1)[0]))

        # Populate a dictionary with these new states
        statesDict = {}
        for i in openStates:
            statesDict[str(i)] = 0
        for j in closedStates:
            statesDict[str(j)] = 1

        # Set the state_dict attribute to this new dictionary
        self.state_dict = statesDict

    def addCycles(self, num_cycles):
        """

            Add number of cycles to network, given randomised states. Only adds cycles between opposite states

        """
        states = self.state_dict
        openStates = dict(filter(lambda elem: elem[1] == 0, states.items()))
        closedStates = dict(filter(lambda elem: elem[1] == 1, states.items()))
        for _ in range(num_cycles):
            possible = possibleCycle(self.adjMatrix, openStates, closedStates)
            if possible[0]:
                self.adjMatrix = createCycle(self.adjMatrix, possible[1])

    def generateAdjMatrix(self):

        _non_zeros = self.trans_matrix != 0
        newMatrix = self.adjMatrix.copy()
        newMatrix[_non_zeros] = 1
        return newMatrix

    def randomiseAll(self, mag=1, num_cycles=0):
        self.randomiseAdj()
        self.randomiseWeights(mag)
        self.randomiseStates()
        self.addCycles(num_cycles)
        return self

    def generateCannonical(self):

        # Partition network into submatrixes by state:
        state_dict = list(self.state_dict.values())
        trans_matrix = self.trans_matrix

        _reference_dict = defaultdict(list)

        for i, value in enumerate(state_dict):
            _reference_dict[value].append(i)

        rearrangeMap = []
        partitions = []
        for (i, j) in sorted(_reference_dict.items(), key=lambda item: item[0]):
            j.sort(key=lambda item: trans_matrix[item, item])

            # Sort out partitioned matrix for diagonalisation
            newPartition = np.array([trans_matrix[m, n] for m, n in it.product(
                j, repeat=2)]).reshape((len(j), len(j)))
            partitions.append(newPartition)

            # Sort out rearranged matrix for final product
            rearrangeMap += j

        new_trans_matrix = np.array([trans_matrix[m, n] for m, n in it.product(
            rearrangeMap, repeat=2)]).reshape((len(trans_matrix), len(trans_matrix)))

        diagonaliser = reduce(directSum, map(getDiagonaliser, partitions))

        newNewTrans = zeroSmalls(np.linalg.inv(
            diagonaliser) @ new_trans_matrix @ diagonaliser)
        return Network(states=newNewTrans.shape[0], trans_matrix=newNewTrans, state_dict={f'{str(v)}': self.state_dict[list(self.state_dict.keys())[v]] for v in rearrangeMap})

    # From http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

    def generateGraph(self, filename):
        """

        Graph generation using NetworkX, outputting a dot file that can be used in GraphViz

        Args:
            filename (string): File name for output file

        Returns:
            bool: Returns True if function completed
        """

        model_df = pd.DataFrame(
            self.trans_matrix, columns=self.state_dict.keys(), index=self.state_dict.keys())

        edges_wts = _get_markov_edges(model_df)

        # create graph object
        G = nx.MultiDiGraph()

        # nodes correspond to states
        G.add_nodes_from(self.state_dict.keys())

        # edges represent transition probabilities
        for k, v in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]
            if v != 0:
                G.add_edge(tmp_origin, tmp_destination,
                           weight=np.round(v, 2), label=np.round(v, 2))

        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)

        # create edge labels for jupyter plot but is not necessary
        edge_labels = {(n1, n2): d['label']
                       for n1, n2, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.drawing.nx_pydot.write_dot(G, f'{filename}.dot')

        return True


class MarkovLog:

    """
        Class for generating real data from a transition rate matrix, or Network model defined in schemaGen.py.
    """

    def __init__(self, Network):
        self.network = Network

        self.time = None
        self.sample_rate = None
        self.noise = None

        self.discrete_history = None
        self.continuous_history = None
        self.data_graph = None
        self.dwell_time_graph = None

        self.analysis_complete = False

    def simulate_discrete(self, time):
        if self.network.trans_matrix is None or self.network.state_dict is None:
            raise TypeError(
                'No model loaded. Please load a model using the load_from_network or load_from_csv methods.')
        else:
            self.time = time
            # Using native lists for increased performance
            histList = []
            states_keys = list(self.network.state_dict.keys())
            states_values = list(self.network.state_dict.values())

            # Randomly select first state
            current_state = random.randint(
                0, len(self.network.trans_matrix) - 1)
            clock = 0
            with tqdm(total=time) as pbar:
                while clock < time:
                    # Sample transitions
                    sojourn_times = [sample_from_rate(
                        rate) for rate in self.network.trans_matrix[current_state]]
                    # Identify next state
                    next_state_index = min(
                        range(len(self.network.trans_matrix)), key=lambda x: sojourn_times[x])

                    # Add histories
                    sojourn_time = sojourn_times[next_state_index]
                    histList.append(
                        [states_keys[current_state], states_values[current_state], sojourn_time])
                    # Advance clock
                    clock += sojourn_time

                    # Update progress bar
                    cur_perc = clock
                    pbar.update(cur_perc - pbar.n)

                    # Set the current state to the next state and restart loop
                    current_state = next_state_index

            self.discrete_history = pd.DataFrame(
                histList, columns=['State', 'Channels', 'Time Spent'])

            return self

    def simulate_continuous(self, sample_rate, noise, **kwargs):

        # Check to see if we have a discrete history
        if self.discrete_history is None:
            # If not, try and generate one using a time keyword arguement.
            if 'time' not in kwargs:
                raise ValueError(
                    'If running a continuous simulation before a discrete simulation, please add a "time" arguement')
            else:
                print('No discrete history found, generating now')
                self.simulate_discrete(time=kwargs['time'])

        self.sample_rate = sample_rate
        self.noise = noise

        # Interpolation stage
        ctsHistory = []
        currentTime = 0
        increment = 1/sample_rate
        # Iterate through the event history and stitch together arrays with size proportional to the time spent on each state
        # TQDM included since this can take some time. Progress bars!
        print("Converting event list into continuous channel data \n")
        pythonCmcHistoryList = self.discrete_history.drop(['Time Spent'], axis=1).values.tolist()
        time_spent = self.discrete_history[['Time Spent']].values.tolist()

        for row, time in tqdm(zip(pythonCmcHistoryList, time_spent)):
            numberSamples = round(time[0] * sample_rate)
            for _ in range(numberSamples):
                ctsHistory.append([*row, currentTime])
                currentTime += increment

        ctsHistory = np.array(ctsHistory)
        print("Continuous simulation Complete")
        # Clean up and give both numpy and pandas formats
        column_names = ['State', 'Channels', *[f'fwd_bwk_{i}' for i in self.network.state_dict.keys()], 'Viterbi', 'Time'] if self.analysis_complete else ['State', 'Channels', 'Time']
        ctsHistoryDF = pd.DataFrame(ctsHistory, columns=column_names)
        # Adding noise to data
        print("Adding noise to current data")
        noisy = self.noise.make_noisy(ctsHistory)
        ctsHistoryDF["Noisy Current"] = noisy
        self.continuous_history = ctsHistoryDF
        return self

    def viterbi_analysis(self):
        # Not very OOP. Refactor?
        emission = np.transpose(generate_emission_matrix(self)).tolist()
        obs = self.discrete_history['Time Spent'].values.tolist()
        states = [i for i in range(len(self.network.state_dict.values()))]
        trans_matrix = normalise_trans_matrix(self.network.trans_matrix).tolist()
        indexes = list(enumerate(self.network.state_dict.items()))
        end_st = int(list(filter(lambda x: x[1][0] == self.discrete_history['State'].values[-1], indexes))[0][0])
        first_state = list(filter(lambda x: x[1][0] == self.discrete_history['State'].values[0], indexes))[0][0]
        a = normalise_trans_matrix(self.network.trans_matrix)

        pi = np.zeros(len(trans_matrix))
        pi[first_state] = 1
        fwd_bwk = forward_backward(self.discrete_history['Time Spent'].values, a, np.array(emission), pi)
        viterbi_hist = viterbi(self.discrete_history['Time Spent'].values, a, np.array(emission), pi)

        self.discrete_history = self.discrete_history.join(pd.DataFrame(fwd_bwk, columns=[f'fwd_bwk_{i}' for i in self.network.state_dict.keys()]), rsuffix='fwd_bwk_state')
        self.discrete_history['Viterbi'] = viterbi_hist
        print(self.discrete_history)
        self.analysis_complete = True

    def viterbi_comparison_graph(self, length):
        if not self.analysis_complete:
            raise ValueError('No Viterbi analysis found, please run the viterbi analysis method!')
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        LENNY = int(length * self.sample_rate)
        truncated_history_df = self.continuous_history[:LENNY]
        ax1.plot(truncated_history_df['Time'], truncated_history_df['Noisy Current'])
        ax1.set_xlabel('Time (secs)')
        ax1.set_ylabel('Current (nA)')
        ax1.set_xticks(np.linspace(0, LENNY, 11))
        ax1.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))


        states = list(map(float, [list(self.network.state_dict.keys()).index(i) for i in truncated_history_df['State']]))
        ax2.plot(truncated_history_df['Time'], truncated_history_df['Viterbi'].values.astype(float), drawstyle='steps-mid', linestyle='--', alpha=0.8, label='Viterbi')
        ax2.plot(truncated_history_df['Time'], states, drawstyle='steps-mid', linestyle=':', alpha=0.8, label='Simulation')
        ax2.set_xticks(np.linspace(0, LENNY, 11))
        ax2.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))

        ax2.set_xlabel('Time (secs)')
        ax2.set_ylabel('State')
        ax2.legend()

        a = truncated_history_df[[f'fwd_bwk_{i}' for i in self.network.state_dict.keys()]].values.astype(float).T
        ax3.imshow(a, aspect='auto', interpolation='none', cmap='Reds')
        
        ax3.set_xticks(np.linspace(0, LENNY, 11))
        ax3.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))

        ax3.set_xlabel('Time (secs)')
        ax3.set_ylabel('State')
        ax3.invert_yaxis()

        plt.tight_layout()
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)



        

    def sample_data_graph(self, length, **kwargs):

        # Check to see if we have a continuous history
        if self.continuous_history is None:
            # If not, try and generate one using keyword arguements.
            if not all(x in kwargs for x in ['sample_rate', 'noise']):
                raise ValueError(
                    'To get a sample data graph, a continuous data history needs to exist first. This cannot be done without sample_rate and noise keywords')
            else:
                print('No continuous history found, Attempting to generate one now:')
                if 'time' in kwargs:
                    self.simulate_continuous(sample_rate=kwargs['sample_rate'], noise=kwargs['noise'], time=kwargs['time'])
                else:
                    self.simulate_continuous(sample_rate=kwargs['sample_rate'], noise=kwargs['noise'])

        LENNY = int(length * self.sample_rate)

        # Truncate continuous history dataframe for performance
        truncctsHistoryDF = self.continuous_history[:LENNY]

        # Graph outputs
        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(truncctsHistoryDF['Time'], truncctsHistoryDF['Noisy Current'],
                alpha=0.75, color='grey', ds="steps-mid")
        ax.set_xlabel('Time (secs)')
        ax.set_ylabel('Current (nA)')
        ax.set_xticks(np.linspace(0, LENNY, 11))
        ax.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))
        ax2 = ax.twinx()
        ax2.set_ylabel('Channels Open')
        ax2.set_xticks(np.linspace(0, LENNY, 11, endpoint=True))
        ax2.set_xticklabels(np.round(np.linspace(
            0, length, 11, endpoint=True), ceil(np.log10(length)) + 2))
        ax2.set_ylim((-1, np.max(truncctsHistoryDF['Channels'].astype(
            'float')) + 1))
        ax2.set_yticks(
            range(np.max(truncctsHistoryDF['Channels'].astype('int')) + 1))
        labels = [str(i) for i in range(
            np.max(truncctsHistoryDF['Channels'].astype('int')) + 1)]
        ax2.set_yticklabels(labels)
        # Plot the number of channels open vs the time. Matplotlib doesn't let us do this with lines, so we have to use a dodgy scatter plot
        sc = ax2.scatter(truncctsHistoryDF['Time'], truncctsHistoryDF['Channels'].astype(
            'float'), c=truncctsHistoryDF['State'].astype('category').cat.codes, s=5, marker="|")
        # Legend comprimise
        def lp(i, j): return plt.plot([], color=sc.cmap(
            sc.norm(i)), mec="none", label=j, ls="", marker="o")[0]
        handles = [lp(i, j) for i, j in enumerate(
            np.unique(truncctsHistoryDF['State']))]

        ax.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        plt.legend(handles=handles)
        plt.tight_layout()
        self.data_graph = fig
        return self

    def dwellTimeGraph(self, **kwargs):
         # Check to see if we have a continuous history
        if not isinstance(self.discrete_history, type(pd.DataFrame())):
            # If not, try and generate one using keyword arguements.
            if not 'time' in kwargs:
                raise ValueError(
                    'To get a dwell time graph, a discrete data history needs to exist first. This cannot be done without the time arguement')
            else:
                print('No discrete history found, Attempting to generate one now:')
                self.simulate_discrete(time=kwargs['time'])

        print("Processing dwells")
        # Note: Only works for open/closed datasets

        openDwells = []
        closedDwells = []
        clutch = 0
        for row in self.discrete_history.to_numpy():
            if row[1] == 0:
                clutch += row[2]
            elif clutch > 0:
                openDwells.append(clutch)
                clutch = 0

        clutch = 0
        for row in self.discrete_history.to_numpy():
            if row[1] == 1:
                clutch += row[2]
            elif clutch > 0:
                closedDwells.append(clutch)
                clutch = 0

        openDwells = np.asarray(openDwells)
        closedDwells = np.asarray(closedDwells)

        f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        maxb = np.max([np.max(openDwells), np.max(closedDwells)])
        minb = np.min([np.min(openDwells), np.min(closedDwells)])
        ax1.hist(openDwells, bins=np.logspace(np.log10(minb),
                                              np.log10(maxb)), color='Green', label="Open Dwell Times")
        ax2.hist(closedDwells, bins=np.logspace(np.log10(minb), np.log10(
            maxb)), color='Red', label="Closed Dwell Times")
        ax1.set_xlabel('Log Time')
        ax1.set_xscale('log')
        ax1.set_yscale('squareroot')
        ax1.set_ylabel('Sqrt DTF')
        ax2.set_xlabel('Log Time')
        ax2.set_xscale('log')
        ax2.set_yscale('squareroot')
        ax2.set_ylabel('Sqrt DTF')
        plt.tight_layout()
        ax1.legend()
        ax2.legend()
        self.dwell_time_graph = f
        return (openDwells, closedDwells, f)


class Noise:
    """
        Class for defining a noise pipeline for adding to a continuous time simulation.

        Each noise layer should return a function that has inputs of the ctsHistory numpy array 
        (See above) and a current array outputted from the previous layer. 
        
        Each noise layer needs to output a current of the same size as the input current, 
        with whatever operations added modifying it.
    """

    def __init__(self):
        self.sequence = []

    def make_noisy(self, array, channels_index=1):
        out = array[:, channels_index].astype('float')
        for layer in self.sequence:
            out = layer(array, out)
        return out

    def add(self, noise_layer):
        self.sequence.append(noise_layer)



# Noise layers - might change this to classes later on... not sure.


def simple_f_noise(exponent, mean=0, sd=1):
    def outfunc(array, current):
        return current + powerlaw_psd_gaussian(exponent, current.shape[0]) * sd + mean
    return outfunc


def scaled_f_noise(exponent, scale_factor=2, mean=0, base_sd=1, channels_index=1):
    def outfunc(array, current):
        modifier = (scale_factor - 1) * \
            array[:, channels_index].astype('float') + 1
        return current + (base_sd * modifier) * powerlaw_psd_gaussian(exponent, current.shape[0]) + mean
    return outfunc


def sinusoidal_noise(amplitude, frequency, time_index=2):
    def outfunc(array, current):
        return current + amplitude * np.sin(2 * np.pi * frequency * array[:, time_index].astype('float'))
    return outfunc


def relaxation_noise_opens(decay_speed, decay_factor, channels_index=1):
    def outfunc(array, current):
        counts = 0
        for i in range(len(array)):
            if int(array[i,channels_index]) == 1:
                current[i] += decay_factor * (np.exp(-decay_speed * counts) - 1)
                counts += 1
            else:
                counts = 0
        return current
    return outfunc