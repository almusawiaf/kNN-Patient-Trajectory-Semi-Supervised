import numpy as np
import matplotlib.pyplot as plt

from dtw import *
from copy import deepcopy


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


# List 1
idx1 = np.linspace(0, 6.28, num = 50)
query = np.sin(idx1) + np.random.uniform(size = 50)/10.0
print (query)

# List 2 [USE any one of the options]
# Option 1.
# idx2 = np.linspace(0, 6.28, num = 100)
# template = np.cos(idx2)
# print (template)

# Option 2.
idx2 = np.linspace(0, 6.28, num = 50)
template = np.sin(idx2) + np.random.uniform(size = 50)/10.0 + 1
print (template)

# Define the distance metric (e.g., Euclidean distance)
dist = lambda x, y: np.abs(x - y)

# Find the best match with the canonical recursion formula
alignment = dtw(query, template, dist)
cost, _, _, (ind1, ind2) = deepcopy(alignment)
print (cost)
print (ind1)
print (ind2)

viz(query, template, ind1, ind2, 'DTW_Render.png')

