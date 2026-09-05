import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        plt.plot([i, j], [Q[i], T[j]], 'k-', alpha=0.1)

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


def remap_code(D):

    Map, ind = {}, 0
    E = []
    for i in range(len(D)):
        if D.iloc[i]['ICD9_CODE'] not in Map.keys():
            Map[D.iloc[i]['ICD9_CODE']] = ind
            ind = ind + 1

        E.append([D.iloc[i]['SUBJECT_ID'], Map[D.iloc[i]['ICD9_CODE']]])

    return E, Map


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


'''
# Parameters
how_many = 50000

# Read data
D = pd.read_csv('DIAGNOSES_ICD.csv')
D, Map = remap_code(D[:how_many])

# Create a dictionary of diagnosis for each patient
P = {}
for [id, code] in D:
    if id not in P.keys():
        P[id] = []
    if code not in P[id]:
        P[id].append(code)
# print (P)

# Visualize similarity between patient pairs
Y, note, max_sim = [], [], 0
Keys = list(P.keys())
for i in range(len(Keys) - 1):
    for j in range(i + 1, len(Keys)):
        l1 = P[Keys[i]]
        l2 = P[Keys[j]]
        JS = jaccard_similarity(l1, l2)
        Y.append(JS)

        if JS > max_sim and min(len(l1), len(l2)) > 25:
            max_sim = JS
            note = deepcopy([i, j])

print (max_sim)

# plt.hist(Y, bins = 50)
# plt.show()

# Find the best match with the canonical recursion formula
query = P[Keys[note[0]]]
template = P[Keys[note[1]]]
'''

# Define the distance metric (e.g., Euclidean distance)
# dist = lambda x, y: np.abs(x - y)
dist = lambda x, y: 0 if x == y else 1

query = [1, 2, 5, 9]
template = [1, 2, 5, 9]

alignment = dtw(query, template, dist)
cost, _, _, (ind1, ind2) = deepcopy(alignment)
print (query)
print (template)
print (cost)

viz(query, template, ind1, ind2, 'DTW_Render.png')
