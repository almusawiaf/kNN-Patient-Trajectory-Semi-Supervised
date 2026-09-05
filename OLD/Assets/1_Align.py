import numpy as np

from copy import deepcopy


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
        start = max(D[p[0]]) + 1
        end   = min(D[p[1]])
        M[p].extend([r for r in range(start, end)])
    return M


# Example usage:
M_1 = {(1, 2): []}
D_1_2 = {1: [2], 2: [5]}
M_1 = compute_missing_ranges(deepcopy(M_1), deepcopy(D_1_2))

print("Updated M:", M_1)
