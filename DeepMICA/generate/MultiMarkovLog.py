import pandas as pd
import numpy as np

from tqdm import tqdm


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


class MultiMarkovLog(networks):

    def __init__(self, networks):

        # Network objects that the log is based on
        self.networks = networks

        # Parameters for simulation
        self.time = None
        self.sample_rate = None
        self.noise = None

        # Actual data
        self.discrete_histories = None
        self.continuous_histories = None
        self.continuous_aggregate = None
        self.data_graph = None
        self.dwell_time_graph = None

    def simulate_discrete(self, time):
        """
            Simulate the multichannel log as independent Markov channels.
        """

        self.time = time

        # Create an empty list for the discrete histories to go into.
        discrete_histories = []

        # Process is very similar to MarkovLog process, just iterating over each network
        for network in self.networks:

            # List of events to be populated below
            history_list = []

            # Get the keys and values of the network
            states_keys = list(network.state_dict.keys())
            states_values = list(network.state_dict.values())

            # Randomly generated the starting state (not great!)
            current_state = random.randint(0, len(network.trans_matrix) - 1)

            # Clock for limiting simulation to time attribute.
            clock = 0

            # Start the discrete simulation with TQDM progress bar
            with tqdm(total=time) as progress_bar:

                while clock < time:

                    # Sample the transitions to get the sojourn times.
                    sojurn_times = [sample_from_rate(rate)
                                    for rate in network.trans_matrix[current_state]]

                    # Get the indx for the next state by taking the minimum sojurn times
                    next_state_index = min(range(len(network.trans_matrix)),
                                           key=lambda x: sojurn_times[x])

                    # Get the lowest sojurn time
                    sojurn_time = sojurn_times[next_state_index]

                    # Append this time to the history list
                    history_list.append([states_keys[current_state],
                                         states_values[current_state],
                                         sojurn_time])

                    # Advance the clock by the sojurn time
                    clock += sojurn_time

                    # Update the progress bar
                    current_percentage = clock
                    pbar.update(current_percentage - pbar.n)

                    # Set the current state to the next state and restart loop
                    current_state = next_state_index

            # Create a dataframe with the discrete history list
            discrete_history = pd.DataFrame(history_list,
                                            columns=["State", "Channels", "Time Spent"])

            # Add this history to list of all histories
            discrete_histories.append(discrete_history)

        # Set attribute
        self.discrete_history = discrete_histories

        return self

        def simulate_continuous(self, sample_rate, noise, **kwargs):

            self.sample_rate = sample_rate
            self.noise = noise

            # Same as standard MarkovLog, but we save the Noise adding till the end
            for idx, discrete_history in enumerate(self.discrete_history):

                # Initialise some variables
                continuous_history = []
                current_time = 0
                increment = 1/sample_rate

                # Iterate through the event history and join arrays with size proportional to sojurn times
                python_cmc_history_list = discrete_history.drop(
                    ["Time Spent"], axis=1).values.tolist()
                time_spent = discrete_history[["Time Spent"]].values.tolist()

                for row, time in tqdm(zip(python_cmc_history_list, time_spent)):
                    number_samples = round(time[0] * sample_rate)

                    for _ in range(number_samples):
                        continuous_history.append([*row, current_time])
                        current_time += increment

                    continuous_history = np.array(continuous_history)

                    # Turn into pandas dataframe and add to induvidual histories list
                    column_names = ['State', 'Channels', 'Time']
                    continuous_history_df = pd.DataFrame(
                        continuous_history, columns=column_names)

                    self.continuous_histories.append(continuous_history_df)

                    # Add the signal to the aggregate signal - note this is still without noise
                    if self.continuous_aggregate == None:
                        # Initialise aggregate history
                        self.continuous_aggregate = pd.DataFrame(continuous_history, columns=[
                                                                 f'State of Channel {idx}', 'Channels', 'Time'])
                    else:
                        # Otherwise add a new column for the new state and add the channels on
                        self.continuous_aggregate[f'State of Channel {idx}'] = continuous_history_df[[
                            'State']]
                        self.continuous_aggregate[[
                            'Channels']] += continuous_history_df[['Channels']]

            # Make the aggregate signal noisy (bit of a hack here)
            noisy = self.noise.make_noisy(
                self.continuous_aggregate[['State of Channel 0', 'Channels', 'Time']].to_numpy())

            self.continuous_aggregate["Noisy Current"] = noisy
            return self

    def sample_data_graph(self, length, **kwargs):

        # Number of points to plot
        lenny = int(length * self.sample_rate)

        # Truncate the dataframe to only include points of interest
        truncated_history_df = self.continuous_aggregate[:lenny]

        # Graph time
        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(truncated_history_df['Time'], truncated_history_df['Noisy Current'],
                alpha=0.75, color='grey', ds="steps-mid")

        ax.set_xlabel("Time (secs)")
        ax.set_ylabel("Current (nA)")
        ax.set_xticks(np.linspace(0, lenny, 11))
        ax.set_xticklabels(np.round(np.linspace(
            0, length, 11), ceil(np.log10(length)) + 2))

        ax2 = ax.twinx()
        ax2.set_tlabel("Channels Open")
        ax2.set_xticks(np.linspace(0, lenny, 11, endpoint=True))
        ax2.set_xticklabels(np.round(np.linspace(
            0, length, 11, endpoint=True), ceil(np.log10(length)) + 2))
        ax2.set_ylim(
            (-1, np.max(truncated_history_df["Channels"].astype('float')) + 1))
        ax2.set_yticks(
            range(np.max(truncated_history_df['Channels'].astype('int')) + 1))

        # Plot the number of channels open vs the time. Matplotlib doesn't let us do this with lines, so we have to use a dodgy scatter plot
        sc = ax2.scatter(truncated_history_df['Time'], truncated_history_df['Channels'].astype(
            'float'), s=5, marker="|")

        ax.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        self.data_graph = fig
        return self
