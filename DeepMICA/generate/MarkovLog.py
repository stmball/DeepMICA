import random

import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

import pandas as pd
import numpy as np


from tqdm import tqdm
from .fwd_bkw import *


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
        pythonCmcHistoryList = self.discrete_history.drop(
            ['Time Spent'], axis=1).values.tolist()
        time_spent = self.discrete_history[['Time Spent']].values.tolist()

        for row, time in tqdm(zip(pythonCmcHistoryList, time_spent)):
            numberSamples = round(time[0] * sample_rate)
            for _ in range(numberSamples):
                ctsHistory.append([*row, currentTime])
                currentTime += increment

        ctsHistory = np.array(ctsHistory)
        print("Continuous simulation Complete")
        # Clean up and give both numpy and pandas formats
        column_names = ['State', 'Channels', *[f'fwd_bwk_{i}' for i in self.network.state_dict.keys(
        )], 'Viterbi', 'Time'] if self.analysis_complete else ['State', 'Channels', 'Time']
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
        trans_matrix = normalise_trans_matrix(
            self.network.trans_matrix).tolist()
        indexes = list(enumerate(self.network.state_dict.items()))
        end_st = int(list(filter(
            lambda x: x[1][0] == self.discrete_history['State'].values[-1], indexes))[0][0])
        first_state = list(filter(
            lambda x: x[1][0] == self.discrete_history['State'].values[0], indexes))[0][0]
        a = normalise_trans_matrix(self.network.trans_matrix)

        pi = np.zeros(len(trans_matrix))
        pi[first_state] = 1
        fwd_bwk = forward_backward(
            self.discrete_history['Time Spent'].values, a, np.array(emission), pi)
        viterbi_hist = viterbi(
            self.discrete_history['Time Spent'].values, a, np.array(emission), pi)

        self.discrete_history = self.discrete_history.join(pd.DataFrame(fwd_bwk, columns=[
                                                           f'fwd_bwk_{i}' for i in self.network.state_dict.keys()]), rsuffix='fwd_bwk_state')
        self.discrete_history['Viterbi'] = viterbi_hist
        print(self.discrete_history)
        self.analysis_complete = True

    def viterbi_comparison_graph(self, length):
        if not self.analysis_complete:
            raise ValueError(
                'No Viterbi analysis found, please run the viterbi analysis method!')

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        LENNY = int(length * self.sample_rate)
        truncated_history_df = self.continuous_history[:LENNY]
        ax1.plot(truncated_history_df['Time'],
                 truncated_history_df['Noisy Current'])
        ax1.set_xlabel('Time (secs)')
        ax1.set_ylabel('Current (nA)')
        ax1.set_xticks(np.linspace(0, LENNY, 11))
        ax1.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))

        states = list(map(float, [list(self.network.state_dict.keys()).index(
            i) for i in truncated_history_df['State']]))
        ax2.plot(truncated_history_df['Time'], truncated_history_df['Viterbi'].values.astype(
            float), drawstyle='steps-mid', linestyle='--', alpha=0.8, label='Viterbi')
        ax2.plot(truncated_history_df['Time'], states, drawstyle='steps-mid',
                 linestyle=':', alpha=0.8, label='Simulation')
        ax2.set_xticks(np.linspace(0, LENNY, 11))
        ax2.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))

        ax2.set_xlabel('Time (secs)')
        ax2.set_ylabel('State')
        ax2.legend()

        a = truncated_history_df[[
            f'fwd_bwk_{i}' for i in self.network.state_dict.keys()]].values.astype(float).T
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
                    self.simulate_continuous(
                        sample_rate=kwargs['sample_rate'], noise=kwargs['noise'], time=kwargs['time'])
                else:
                    self.simulate_continuous(
                        sample_rate=kwargs['sample_rate'], noise=kwargs['noise'])

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
