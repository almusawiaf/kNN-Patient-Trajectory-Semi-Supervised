import numpy as np
import matplotlib.pyplot as plt

from dtw import *
from copy import deepcopy


def maps(l1, l2):

    D = {}
    for i in range(len(l1)):
        if l1[i] not in D.keys():
            D[l1[i]] = []

        D[l1[i]].append(l2[i])

    return D


def viz(Q, T, ind1, ind2, fname):
    # Convert to numpy arrays for ease of indexing
    Q = np.array(Q)
    T = np.array(T)

    # Plot the query and template signals with alignment lines
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(Q)), Q, 'r-', label='Query Signal')
    plt.plot(range(len(T)), T, 'b-', label='Template Signal')

    # Plot the alignment lines
    for i, j in zip(ind1, ind2):
        plt.plot([i, j], [Q[i], T[j]], 'k-', alpha=0.5)

    plt.scatter(ind1, Q[ind1], color='red')
    plt.scatter(ind2, T[ind2], color='blue')

    # Labeling the plot
    plt.xlabel('Index')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.title('DTW Alignment')

    plt.tight_layout()
    plt.savefig(fname, dpi = 300)
    plt.show()


def compute_missing_ranges(M, D):
    """
    Compute the missing ranges in M based on D.

    Parameters:
    - M: Dictionary where keys are tuples representing pairs of indices and values are lists to store ranges.
    - D: Dictionary where keys are indices and values are lists of indices from another list.

    Returns:
    - Updated M dictionary with computed ranges appended to each key's list.
    """

    for p in M:

        start = min(D[p[0]]) + 1
        end = max(D[p[1]])
        M[p].extend([r for r in range(start, end)])

    return M


# # Example usage:
# M_1 = {(1, 2): []}
# D_1_2 = {1: [2], 2: [5]}

# List 1
idx1 = np.linspace(0, 6.28, num = 20)
query = np.sin(idx1) + np.random.uniform(size = 20)/10.0

# List 2 [USE any one of the options]
# Option 1.
# idx2 = np.linspace(0, 6.28, num = 100)
# template = np.cos(idx2)
# print (template)

# Option 2.
idx2 = np.linspace(0, 6.28, num = 10)
template = np.sin(idx2) + np.random.uniform(size = 10)/10.0 + 1

# Define the distance metric (e.g., Euclidean distance)
dist = lambda x, y: 0 if x == y else 1

# Find the best match with the canonical recursion formula
alignment = dtw(query, template, dist)
cost, _, _, (query_ind, template_ind) = deepcopy(alignment)
print (query_ind)
print (template_ind)

dic_qt = maps(query_ind, template_ind)
print (dic_qt)

dic_tq = maps(template_ind, query_ind)
print (dic_tq)

# # Example usage:
M_1 = {(0, 1): []}
# D_1_2 = {1: [2], 2: [5]}

M_1 = compute_missing_ranges(deepcopy(M_1), deepcopy(dic_tq))
print("Updated M:", M_1)

viz(query, template, query_ind, template_ind, 'DTW_Render.png')