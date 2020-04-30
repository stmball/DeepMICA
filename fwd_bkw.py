import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from time import sleep

def dict_norm(dicto):
    out = {}
    norm = sum(dicto.values())
    for key, value in dicto.items():
        out[key] = value / norm

    return out



def calculate_recurrrent_term(new_state_index, last_state_probs, trans_matrix):
    """
        Function for calculating the recurrent term in the emission matrix process

        Arguements:

        new_state_index (int): index of the new state we are looking for.
        last_state_probs (list): list of the previous state probabilities
        trans_matrix (np.array): transition matrix 

        Returns:

        prob (float): Probability
    """

    return sum([trans_matrix[i, new_state_index] * last_state_probs[0, i] if i != new_state_index else 0 for i in range(trans_matrix.shape[0])])


def calculate_other_terms(pi, q_square, q_not_square, t_c):
    """
        Function for calculating the other terms in the emission matrix process.
        For the numerator term pi should be set to a matrix with length trans_matrix.shape[0]
        with a 1 in the sth index and 0s otherwise.
        For the denominator term pi should be the vector of probabiliies of being in each state (use above function)
    """

    a = pi
    a = a @ np.linalg.inv(q_square)
    a = a @ (expm(q_square * t_c) - np.identity(q_square.shape[0]))
    a = a @ q_not_square
    a = a @ np.ones((1, q_not_square.shape[1])).T
    return 1 - a
    # return pi @ np.linalg.inv(q_square) @ (np.exp(q_square * t_c) @ q_not_square @ np.ones((1, q_not_square.shape[1]))).T

def normalise_trans_matrix(trans_matrix):
    # Transform trans_matrix into normalised form - rows sum to one.
    new_trans = np.copy(trans_matrix).astype('float64')
    for idx, row in enumerate(new_trans):
        for idy, col in enumerate(row):
            if col < 0:
                new_trans[idx, idy] = 0
        a = row / row.sum(0)
        new_trans[idx, :] = a

    return new_trans

def generate_emission_matrix(log):
    # Get important values from MarkovLog object
    trans_matrix = log.network.trans_matrix
    state_dict = log.network.state_dict
    discrete_history = log.discrete_history

    # Get open and closed state indexes
    indexes = list(enumerate(state_dict.items()))
    opens = list(filter(lambda x: x[1][1] == 0, indexes))
    closeds = list(filter(lambda x: x[1][1] == 1, indexes))

    # Split into matricies
    q_oo = np.zeros((len(opens), len(opens)))
    q_cc = np.zeros((len(closeds), len(closeds)))
    q_co = np.zeros((len(closeds), len(opens)))
    q_oc = np.zeros((len(opens), len(closeds)))

    # Transform trans_matrix into normalised form - rows sum to one.
    new_trans = normalise_trans_matrix(trans_matrix)

    # Populate q_oo, q_cc, q_co, q_oc
    for (state_one, state_two, matrix) in [(opens, opens, q_oo), (closeds, closeds, q_cc), (closeds, opens, q_co), (opens, closeds, q_oc)]:
        for idi, i in enumerate(state_one):
            for idj, j in enumerate(state_two):
                matrix[idi, idj] = trans_matrix[i[0], j[0]]

    # Get first state and create a vector with all zeros apart from a 1 where the first state is.
    first_state = list(filter(lambda x: x[1][0] == discrete_history['State'].values[0], indexes))[0][0]
    pi = np.zeros(trans_matrix.shape[0])
    pi[first_state] = 1
    pi = pi.reshape((1, len(pi)))
    probabilities = []
    probabilities.append(pi)
    first_time = True
    for row in tqdm(discrete_history.values):

        row_probs = []
        parity = row[1]
        prev_row_probs = probabilities[-1]
        for i in range(trans_matrix.shape[0]):
            if parity == 0:
                # We are open!
                if i in [j[0] for j in opens]:
                    # Iterate through open states and calculate probability of being here given the observed open dwell time.
                    # First, calculate the recurrent term P(S_t = s).
                    
                    # Find the index in the open states list that corresponds to the current state
                    for idx, open_state in enumerate(opens):
                        if i == open_state[0]:
                            index = idx
                            break
                    
                    if first_time and i == first_state:
                        recurrent_term = 1.0
                        first_time = False
                    else:
                        recurrent_term = calculate_recurrrent_term(
                            i, prev_row_probs, new_trans)

                    # Calculate numerator emission probability P(e_t >= e | S_t = s)
                    num_pi = np.zeros(len(opens))




                    num_pi[index] = 1
                    num_pi = num_pi.reshape((1, len(num_pi)))

                    # Calculate probability
                    num_other = calculate_other_terms(
                        num_pi, q_oo, q_oc, row[2])[0, 0]

                    # Calculate denominator emission probability P(e_t >= e)
                    # Find indexes of probability vector that correspond to open states
                    open_indexes = []
                    for k in opens:
                        if k[1][1] == 0:
                            open_indexes.append(k[0])

                    # Get the vector of previous row probabilies for only open indexes
                    dem_pi = np.array([prev_row_probs[0, p]
                                       for p in open_indexes])
                    dem_pi = np.reshape(dem_pi, (1, len(dem_pi)))

                    # Calculate probability
                    dem_other = calculate_other_terms(
                        dem_pi, q_oo, q_oc, row[2])[0, 0]

                    # Calculate total probability and add to overall event probability vector

                    prob = num_other * recurrent_term / dem_other
                    row_probs.append(prob)

                else:
                    # If the state is closed, we know the probability of closed -> closed is zero for canonical matricies
                    row_probs.append(0)

            else:
                if i in [j[0] for j in closeds]:
                    # Iterate through open states and calculate probability of being here given the observed open dwell time.
                    # First, calculate the recurrent term P(S_t = s).
                    
                    
                    for idx, closed_state in enumerate(closeds):
                        if i == closed_state[0]:
                            index = idx
                            break
                    

                    if first_time and i == first_state:
                        recurrent_term = 1.0
                        first_time = False
                    else:
                        recurrent_term = calculate_recurrrent_term(
                            i, prev_row_probs, new_trans)

                    num_pi = np.zeros(len(closeds))

                    for idx, closed_state in enumerate(closeds):
                        if i == closed_state[0]:
                            index = idx
                            break
                    num_pi[index] = 1

                    num_pi = num_pi.reshape((1, len(num_pi)))
                    num_other = calculate_other_terms(
                        num_pi, q_cc, q_co, row[2])[0, 0]

                    closed_indexes = []
                    for k in closeds:
                        if k[1][1] == 1:
                            closed_indexes.append(k[0])

                    dem_pi = np.array([prev_row_probs[0, p]
                                       for p in closed_indexes])
                    dem_pi = np.reshape(dem_pi, (1, len(dem_pi)))

                    dem_other = calculate_other_terms(
                        dem_pi, q_cc, q_co, row[2])[0, 0]

                    prob = num_other * recurrent_term / dem_other
                    row_probs.append(prob)
                else:
                    row_probs.append(0)

        row_probs = np.array(row_probs)
        row_probs = row_probs / row_probs.sum(0)
        row_probs = np.reshape(row_probs, (1, len(row_probs)))
        probabilities.append(row_probs)
    return np.squeeze(np.array(probabilities)[1:])

# Edited from http://www.adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, 0]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, t]
        alpha[t, :] = alpha[t, :]/np.sum(alpha[t, :])
    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, t + 1]).dot(a[j, :])
        beta[t, :] = beta[t, :]/np.sum(beta[t, :])    
    return beta

def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, 0])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, t])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    return S

def forward_backward(V, a, b, initial_distribution):
    alphas = forward(V, a, b, initial_distribution)
    betas = backward(V, a, b)
    posterior = np.zeros_like(alphas)
    for i in range(len(posterior)):
        posterior[i, :] = alphas[i, :] * betas[i, :]
        posterior[i, :] = posterior[i, :]/np.sum(posterior[i, :])
    
    return posterior