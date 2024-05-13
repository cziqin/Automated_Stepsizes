import numpy as np


def cycle_graph(m):
    # Initialize the matrix with zeros
    W = np.zeros((m, m))

    # Set diagonal elements (self-connections)
    np.fill_diagonal(W, 0.4)

    # Set adjacent connections in a cycle
    for i in range(m):
        W[i, (i + 1) % m] = 0.3  # Connection to next node
        W[i, (i - 1 + m) % m] = 0.3  # Connection to previous node

    return W


def fully_connected_graph(num_agents, weight):
    # Create an array of ones with the shape (num_agents, num_agents)
    W = np.ones((num_agents, num_agents))

    # Multiply the whole matrix by the weight
    W *= weight

    return W


# Number of agents

def generate_Ei_matrix(W, i):
    m = W.shape[0]
    connected_indices = [j for j in range(m) if W[i, j] != 0 and j != i]
    deg_i = len(connected_indices)
    E_i = np.zeros((deg_i + 1, m))
    for idx, j in enumerate(connected_indices):
        E_i[idx, j] = 1
    E_i[-1, i] = 1

    return E_i, deg_i


def distributively_estimate_consensus_params(w, e, deg):
    m = w.shape[0]

    I = np.eye(m)
    z = {}
    s = {}
    Gamma = {}
    Psi = {}
    for i in range(m):
        z[i] = {}
        s[i] = {}
        Gamma[i] = np.zeros((1, (deg + 1) * (m - deg)))
        Psi[i] = np.zeros(((deg + 1) * (m - deg), m))
        for l in range(m):
            z[i][l] = [I[:, l]]
            s[i][l] = [np.dot(e[i], z[i][l][0])]

    for l in range(m):
        for k in range(m - deg):
            for i in range(m):
                summation = np.zeros_like(z[i][l][0])
                for j in range(m):
                    summation += w[i, j] * z[j][l][k]
                z[i][l].append(summation)
                s[i][l].append(np.dot(e[i], z[i][l][k]))

    # Formulate Psi_i matrix

    for i in range(m):
        for k in range(m - deg):
            # Loop through each dimension l
            for l in range(m):
                # Calculate the row offset for placing the data
                row_offset = k * (deg + 1)
                # Each s[i][l][k] is a (deg_i+1)-dimensional vector
                # We place it in rows starting from 'row_offset' to 'row_offset + deg_i + 1' in column l
                Psi[i][row_offset:row_offset + (deg + 1), l] = s[i][l][k]

        Psi_i_pinv = np.linalg.pinv(Psi[i])

        # Compute Gamma_i using the pseudoinverse
        Gamma[i] = (1 / m) * np.ones((1, m)) @ Psi_i_pinv

    return Gamma

#
