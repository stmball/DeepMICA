import numpy as np
import pandas as pd
import networkx as nx
import itertools as it

from collections import defaultdict
from functools import reduce

import random

class Network:

    def __init__(self, states, **kwargs):
        """

        Network object initialisation - can accept a number of arguements for
        predefining a network, or leaving it empty for later population.

        Args:
            states (int): Number of states of the network

        Kwargs:
            adj_matrix (np.darray): Predefined adjacency matrix for network
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
        if 'adj_matrix' in kwargs:
            adj_matrix = kwargs['adj_matrix']
            # Check type is numpy array
            if not isinstance(adj_matrix, np.ndarray):
                raise TypeError('Adjacency matrix must be numpy array.')

            # Check adj_matrix is square with rows and columns equal to the state arguement
            elif adj_matrix.shape != (states, states):
                raise TypeError(
                    'Adjacency matrix dimensions must agree with number of states.')

            # Check for symmetry
            elif np.array_equal(adj_matrix, adj_matrix.T):
                raise TypeError('Adjacency matrix must be symmetric!')

            # Check that matrix has only 0s and 1s
            elif np.any(adj_matrix != 0 and adj_matrix != 1):
                raise TypeError(
                    'Adjacency matrix must only have one and zero entries!')

            # If all tests are clear, set adj_matrix attribute to the kwarg
            else:
                self.adj_matrix = adj_matrix

        # If no adjacency matrix is given, initalise an empty zero adjacency matrix
        else:
            self.adj_matrix = np.zeros((states, states))

        if 'trans_matrix' in kwargs:
            trans_matrix = kwargs['trans_matrix']
            # Check type is numpy array
            if not isinstance(trans_matrix, np.ndarray):
                raise TypeError('Transition matrix must be numpy array.')

            # Check trans_matrix is square with rows and columns equal to the state arguement
            elif trans_matrix.shape != (states, states):
                raise TypeError(
                    'Transition matrix dimensions must agree with number of states.')

            elif not self.check_markov_form(kwargs['trans_matrix']):
                raise ValueError('Transition matrix must be in proper Markov form - entries on the diagonal should equal the negative sum of the other row elements.')
            # If all tests are clear, set trans_matrix attribute to the kwarg
            else:
                self.trans_matrix = trans_matrix
                if 'adj_matrix' in kwargs and self.adj_matrix != self.generate_adj_matrix():
                    raise Warning(
                        'Transition matrix has non zero rates in positions where adjacency matrix has ones.')
                if 'adj_matrix' not in kwargs:
                    self.adj_matrix = self.generate_adj_matrix()

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

    def check_markov_form(self, trans_matrix):
        for idx, row in enumerate(trans_matrix):
            if row[idx] != -sum(np.delete(row, idx)):
                return False
        else:
            return True

    def randomise_adj(self):
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
            self.adj_matrix[joinee, joiner] = 1
            self.adj_matrix[joiner, joinee] = 1
            connectedNodes.append(joinee)
            disconnectedNodes.remove(joinee)

    def randomise_weights(self, mag, preserveCanonical=False):
        """

        Method for randomising the transition rate matrix for a network up to a given magnitude

        Args:
            mag (float): Ceiling value for random transition rate entries

        """

        adj = self.adj_matrix
        # Generate random numpy matrix with maximal possible value of the mag arguement

        if preserveCanonical:
            opens = list(self.state_dict.values()).count(0)
            closes = list(self.state_dict.values()).count(1)
            flag = True
            while flag:
                randoms = np.multiply(adj, np.random.rand(*adj.shape) * mag)
                flag = not self.check_random_canonical(randoms, opens, closes)
        else:
            randoms = np.multiply(adj, np.random.rand(*adj.shape) * mag)
        # Fix the diagonal to be the negative sum of all other entries in that row
        for i in range(randoms.shape[0]):
            randoms[i, i] = -1 * (np.sum(randoms[i, :]) - randoms[i, i])

        # Set new transition matrix
        self.trans_matrix = randoms
        return self

    def randomise_states(self):
        """

        Method for randomising the state natures of a network under the constraint that no two
        connected nodes can be of the same nature.

        """

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
                    set(np.where(self.adj_matrix[node] == 1)[0]))
            for node in closedStates:
                openStates = openStates.union(
                    set(np.where(self.adj_matrix[node] == 1)[0]))

        # Populate a dictionary with these new states
        statesDict = {}
        for i in openStates:
            statesDict[str(i)] = 0
        for j in closedStates:
            statesDict[str(j)] = 1

        # Set the state_dict attribute to this new dictionary
        self.state_dict = statesDict

    def add_cycles(self, num_cycles):
        """

            Add number of cycles to network, given randomised states. Only adds cycles between opposite states

        """
        states = self.state_dict
        openStates = dict(filter(lambda elem: elem[1] == 0, states.items()))
        closedStates = dict(filter(lambda elem: elem[1] == 1, states.items()))
        for _ in range(num_cycles):
            possible = self.possible_cycle(self.adj_matrix, openStates, closedStates)
            if possible[0]:
                self.adj_matrix = self.create_cycle(self.adj_matrix, possible[1])


    def create_cycle(self, network, stateClasses):
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

    def possible_cycle(self, network, openStates, closedStates):
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

    def generate_adj_matrix(self):

        _non_zeros = self.trans_matrix != 0
        newMatrix = self.adj_matrix.copy()
        newMatrix[_non_zeros] = 1
        return newMatrix

    def randomise_all(self, mag=1, num_cycles=0):
        self.randomise_adj()
        self.randomise_weights(mag)
        self.randomise_states()
        self.add_cycles(num_cycles)
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

        diagonaliser = reduce(self.direct_sum, map(self.get_diagonaliser, partitions))

        newNewTrans = self.zero_smalls(np.linalg.inv(
            diagonaliser) @ new_trans_matrix @ diagonaliser)

        return Network(states=newNewTrans.shape[0], trans_matrix=newNewTrans, state_dict={f'{str(v)}': self.state_dict[list(self.state_dict.keys())[v]] for v in rearrangeMap})

    def check_random_canonical(self, randoms, opens, closes):
        for block in [randoms[0:opens, opens:], randoms[opens:, 0:opens]]:
            if not np.array_equal(np.sum(block, axis=0), np.sort(np.sum(block, axis=0))):
                return False
            elif not np.array_equal(np.sum(block, axis=1), np.sort(np.sum(block, axis=1))):
                return False
            else:
                return True


    def direct_sum(self, a, b):
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

    def get_diagonaliser(self, matrix):
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

    def zero_smalls(self, matrix):
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

    def _get_markov_edges(self, Q):
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

        edges_wts = self._get_markov_edges(model_df)

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
