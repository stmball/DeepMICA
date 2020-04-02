import numpy as np

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
    return sum([trans_matrix[i, new_state_index] * last_state_probs[0, i] if i != new_state_index  else 0 for i in range(trans_matrix.shape[0]) ])

def calculate_other_terms(pi, q_square, q_not_square, t_c):

    """
        Function for calculating the other terms in the emission matrix process.
        For the numerator term pi should be set to a matrix with length trans_matrix.shape[0]
        with a 1 in the sth index and 0s otherwise.
        For the denominator term pi should be the vector of probabiliies of being in each state (use above function)
    """
    print(f'pi: {pi}, pi shape: {pi.shape}')
    print(f'q_square: {q_square}, q_square shape: {q_square.shape}')
    print(f'q_not_square: {q_not_square}, q_not_square shape; {q_not_square.shape}')
    print(f't_c: {t_c}')
    print(f'Attempting to multiply shapes {pi.shape} by {q_square.shape} by {q_square.shape} by {q_not_square.shape} by {np.ones((1, q_not_square.shape[1])).T.shape}')
    
    a = pi
    a = a @ np.linalg.inv(q_square)
    a = a @ (np.exp(q_square * t_c) - np.identity(q_square.shape[0]))
    a = a @ q_not_square
    a = a @ np.ones((1, q_not_square.shape[1])).T
    return 1 - a
    #return pi @ np.linalg.inv(q_square) @ (np.exp(q_square * t_c) @ q_not_square @ np.ones((1, q_not_square.shape[1]))).T



def generate_emission_matrix(log):
    trans_matrix = log.network.trans_matrix
    state_dict = log.network.state_dict
    discrete_history = log.discrete_history

    # Get open and closed state indexes
    indexes = list(enumerate(state_dict.items()))
    opens = list(filter(lambda x: x[1][1] == 0, indexes))
    closeds = list(filter(lambda x: x[1][1] == 1, indexes))
    print(indexes)
    print(opens)
    print(closeds)


    # Split into matricies
    q_oo = np.zeros((len(opens), len(opens)))
    q_cc = np.zeros((len(closeds), len(closeds)))
    q_co = np.zeros((len(closeds), len(opens)))
    q_oc = np.zeros((len(opens), len(closeds)))

    # Transform trans_matrix into normalised form
    new_trans = np.copy(trans_matrix).astype('float64')
    for idx, row in enumerate(new_trans):
        for idy, col in enumerate(row):
            if col < 0:
                print('zeroing')
                new_trans[idx, idy] = 0
        a = row / row.sum(0)
        new_trans[idx, :] = a
        
    for (state_one, state_two, matrix) in [(opens, opens, q_oo), (closeds, closeds, q_cc), (closeds, opens, q_co), (opens, closeds, q_oc)]:
        for idi, i in enumerate(state_one):
            for idj, j in enumerate(state_two):
                matrix[idi,idj] = trans_matrix[i[0], j[0]]
    print(q_oo, q_cc, q_co, q_oc)
    # Iterate over each row
    first_state = list(filter(lambda x: x[1][0] == discrete_history['State'].values[0], indexes))[0][0]
    pi = np.zeros(trans_matrix.shape[0])
    print(f' First state: {first_state}')
    pi[first_state] = 1
    probabilities = []
    pi = np.reshape(pi, (1, len(pi)))
    probabilities.append(pi)
    for row in discrete_history.values:
        row_probs = []
        parity = row[1]
        prev_row_probs = probabilities[-1]
        for i in range(trans_matrix.shape[0]):
            if parity == 1:
                if i in [j[0] for j in opens]:
                    recurrent_term = calculate_recurrrent_term(i, prev_row_probs, new_trans)
                    print(f'Recurrent term: {recurrent_term}')
                    num_pi = np.zeros(len(opens))

                    for idx, open_state in enumerate(opens):
                        if i == open_state[0]:
                            index = idx
                            break
                    print(index)

                    num_pi[index] = 1
                    num_pi = num_pi.reshape((1, len(num_pi)))
                    num_other = calculate_other_terms(num_pi, q_oo, q_oc, row[2])[0,0]
                    print(f'Numerator term: {num_other}')

                    open_indexes = []
                    for k in opens:
                        if k[1][1] == 0:
                            open_indexes.append(k[0])
                    print(open_indexes)

                    dem_pi = np.array([prev_row_probs[0,p] for p in open_indexes])
                    dem_pi = np.reshape(dem_pi, (1, len(dem_pi)))
                    print(dem_pi)

                    dem_other = calculate_other_terms(dem_pi, q_oo, q_oc, row[2])[0,0]
                    print(f'Denominator term: {dem_other}')

                    prob = num_other * recurrent_term / dem_other
                    row_probs.append(prob)
                    print(f'\nProbability is: {prob}\n')

                else:
                    row_probs.append(0)
                    print(f'\nProbability is: 0\n')

            else:
                if i in [j[0] for j in closeds]:
                    recurrent_term = calculate_recurrrent_term(i, prev_row_probs, new_trans)
                    print(f'Recurrent term: {recurrent_term}')
                    num_pi = np.zeros(len(closeds))
                    print(i)

                    for idx, closed_state in enumerate(closeds):
                        if i == closed_state[0]:
                            index = idx
                            break
                    print(index)
                    num_pi[index] = 1

                    num_pi = num_pi.reshape((1, len(num_pi)))
                    num_other = calculate_other_terms(num_pi, q_cc, q_co, row[2])[0,0]
                    print(f'Numerator term: {num_other}')


                    closed_indexes = []
                    for k in closeds:
                        if k[1][1] == 1:
                            closed_indexes.append(k[0])
                    print(closed_indexes)

                    dem_pi = np.array([prev_row_probs[0,p] for p in closed_indexes])
                    dem_pi = np.reshape(dem_pi, (1, len(dem_pi)))
                    print(dem_pi)

                    dem_other = calculate_other_terms(dem_pi, q_cc, q_co, row[2])[0,0]
                    print(f'Denominator term: {dem_other}')

                    prob = num_other * recurrent_term / dem_other
                    row_probs.append(prob)
                    print(f'\nProbability is: {prob}\n')
                else:
                    row_probs.append(0)
                    print(f'\nProbability is: 0\n')

        row_probs = np.array(row_probs)
        print(row_probs)
        row_probs = row_probs / row_probs.sum(0)
        print(row_probs)
        row_probs = np.reshape(row_probs, (1, len(row_probs)))
        print(row_probs)
        probabilities.append(row_probs)
        print(f'FINISHED AN ITERATION\n {row_probs} \n \n ')
    return np.array(probabilities)

# From wikipedia!
def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """Forward–backward algorithm."""
    # Forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # Backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior



