import pickle
import random

MASK_TOKEN = "D_X"

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

def save_dict_to_pickle(dictionary, filename):
    import os
    print(f'Saving the dictionary to {filename}...')
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the dictionary to the pickle file
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print('Saving complete...')
    


def remove_and_mask(clinical_events: dict):
    """
    Randomly remove one item from each admission's event list,
    insert a mask token at the same position, and track ground truth.

    Args:
        clinical_events: {admission_id: [event_codes]}
    Returns:
        masked_data:   {admission_id: [event_codes with MASK_TOKEN at removed position]}
        removed_items: {admission_id: removed_event_code}
    """
    masked_data   = {}
    removed_items = {}

    for adm_id, event_list in clinical_events.items():
        if not event_list:
            masked_data[adm_id] = []
            continue

        remove_pos  = random.randint(0, len(event_list) - 1)

        # Store ground truth value only
        removed_items[adm_id] = event_list[remove_pos]

        # Replace removed item with mask token (position preserved by D_X)
        masked_list = event_list.copy()
        masked_list[remove_pos] = MASK_TOKEN
        masked_data[adm_id] = masked_list

    return masked_data, removed_items

# ── Temp Data ─────────────────────────────────────────────────────────────────

# # Creating data of 10 items only :-)

# admissions = list(clinical_events.keys())

# temp_admission = admissions[:10]

# temp_clinical_events = {i: clinical_events[i] for i in temp_admission}


# ── Run ───────────────────────────────────────────────────────────────────────

# random.seed(42)
# # masked_data, removed_items = remove_and_mask(temp_clinical_events)
# masked_data, removed_items = remove_and_mask(clinical_events)

# save_dict_to_pickle(masked_data, '/lustre/home/almusawiaf/PhD_Projects/Satyaki/Semi_Supervised/Data/masked_data.pkl')
# save_dict_to_pickle(removed_items, '/lustre/home/almusawiaf/PhD_Projects/Satyaki/Semi_Supervised/Data/removed_items.pkl')



# ************************************************************************************************************************************
# ************************************************************************************************************************************


import random
import numpy as np
from collections import defaultdict

MASK_TOKEN = "D_X"

# ── Build vocabulary from all sequences ──────────────────────────────────────
def build_vocab(clinical_events: dict):
    """Map every unique event code → integer index (D_X gets index 0)."""
    vocab = {MASK_TOKEN: 0}
    for event_list in clinical_events.values():
        for code in event_list:
            if code not in vocab:
                vocab[code] = len(vocab)
    return vocab

def encode_sequence(seq: list, vocab: dict):
    """Convert a list of event codes → numpy array of ints."""
    return np.array([vocab.get(code, 0) for code in seq], dtype=np.float64)



# ************************************************************************************************************************************
# ************************************************************************************************************************************


import numpy as np

def dtw_distance(seq_a: list, seq_b: list, vocab: dict):
    """
    Pure numpy DTW — no external libraries needed.
    """
    a = encode_sequence(seq_a, vocab)
    b = encode_sequence(seq_b, vocab)

    n, m = len(a), len(b)
    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i-1] - b[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]  # match
            )

    return dtw_matrix[n, m]






# ************************************************************************************************************************************
# ************************************************************************************************************************************

def find_knn(query_seq: list,
             dataset: dict,
             vocab: dict,
             K: int = 3,
             query_id=None):
    """
    Find K nearest neighbors for query_seq from dataset.

    Args:
        query_seq:  sequence with D_X mask token
        dataset:    {adm_id: [event_codes]}  ← full (unmasked) sequences
        vocab:      event → int mapping
        K:          number of neighbors
        query_id:   exclude self if query is already in dataset

    Returns:
        neighbors: [(adm_id, dtw_distance), ...] sorted nearest first
    """
    distances = {}
    for adm_id, seq in dataset.items():
        if adm_id == query_id:
            continue                          # skip self
        distances[adm_id] = dtw_distance(query_seq, seq, vocab)

    # Sort by distance, take top K
    neighbors = sorted(distances.items(), key=lambda x: x[1])[:K]
    return neighbors


# ************************************************************************************************************************************
# ************************************************************************************************************************************


def impute_missing(query_seq: list,
                   neighbors: list,
                   dataset: dict,
                   epsilon: float = 1e-6):
    """
    Impute D_X in query_seq using weighted vote from neighbors.

    Args:
        query_seq:  masked sequence containing D_X
        neighbors:  [(adm_id, dtw_dist), ...] from find_knn()
        dataset:    {adm_id: [event_codes]} full sequences
        epsilon:    smoothing to avoid division by zero

    Returns:
        predicted event code (str)
    """
    mask_pos = query_seq.index(MASK_TOKEN)  # ← find position j of D_X

    vote_weights = defaultdict(float)       # ← accumulates: Σ w_i for each candidate s_{i,j}

    for adm_id, dist in neighbors:          # ← loop over S_i ∈ N_new
        neighbor_seq = dataset[adm_id]

        if mask_pos >= len(neighbor_seq):
            continue

        candidate_code = neighbor_seq[mask_pos]   # ← s_{i,j}  (event at position j in neighbor i)
        w_i = 1.0 / (dist + epsilon)              # ← w_i = 1 / (d(S_new, S_i) + ε)
        vote_weights[candidate_code] += w_i       # ← Σ w_i · s_{i,j}  grouped by code

    predicted = max(vote_weights, key=vote_weights.get)  # ← argmax = ŝ_j
    return predicted

# ************************************************************************************************************************************
# ************************************************************************************************************************************


def hierarchical_update(new_sequences: dict,
                        dataset: dict,
                        vocab: dict,
                        K: int = 3,
                        recompute_threshold: int = 2):
    """
    Update KNN structure as new sequences arrive.

    Args:
        new_sequences:        {adm_id: masked_seq}  ← incoming batch
        dataset:              {adm_id: full_seq}     ← existing reference data
        vocab:                event → int mapping
        K:                    number of neighbors
        recompute_threshold:  F(S_i) cutoff — sequences appearing in ≥ this
                              many new KNNs get their neighbors recomputed

    Returns:
        knn_map:         {new_adm_id: [(neighbor_id, dist), ...]}
        recomputed_ids:  set of existing sequences that were re-evaluated
        frequency:       {adm_id: count} — how often each appeared as neighbor
    """
    knn_map   = {}
    frequency = defaultdict(int)          # F(S_i) from the paper

    # ── Step A: find KNN for every new sequence ───────────────────────────────
    for adm_id, seq in new_sequences.items():
        neighbors = find_knn(seq, dataset, vocab, K=K, query_id=adm_id)
        knn_map[adm_id] = neighbors

        for neighbor_id, _ in neighbors:
            frequency[neighbor_id] += 1   # track how often each is chosen

    # ── Step B: recompute neighbors for high-frequency existing sequences ─────
    # These are the "hub" sequences whose local structure may have shifted
    recomputed_ids = {
        sid for sid, freq in frequency.items()
        if freq >= recompute_threshold
    }

    recomputed_knn = {}
    for sid in recomputed_ids:
        recomputed_knn[sid] = find_knn(dataset[sid], dataset, vocab,
                                       K=K, query_id=sid)

    print(f"\n  Frequency F(S_i) across new sequence KNNs:")
    for sid, freq in sorted(frequency.items(), key=lambda x: -x[1]):
        flag = " ← recomputed" if sid in recomputed_ids else ""
        print(f"    {sid}: appeared {freq}x{flag}")

    return knn_map, recomputed_ids, frequency

# ************************************************************************************************************************************
# ************************************************************************************************************************************


def run_pipeline(clinical_events: dict, K: int = 3, seed: int = 42):

    random.seed(seed)

    # 1. Build vocabulary
    vocab = build_vocab(clinical_events)
    print(f"Vocabulary size: {len(vocab)} unique codes\n")
    print(vocab)

    # 2. Mask one event per admission
    masked_data, removed_items = remove_and_mask(clinical_events)

    # 3. For each masked admission, find KNN and impute
    predictions = {}
    print(f"{'Admission':<12} {'True':<10} {'Predicted':<10} {'Match'}")
    print("-" * 44)

    for adm_id, masked_seq in masked_data.items():

        # Dataset = all OTHER admissions (unmasked as reference)
        reference = {k: v for k, v in clinical_events.items() if k != adm_id}

        neighbors   = find_knn(masked_seq, reference, vocab, K=K)
        predicted   = impute_missing(masked_seq, neighbors, reference)
        predictions[adm_id] = predicted

        true_val = removed_items[adm_id]
        match    = "✅" if predicted == true_val else "❌"
        print(f"{adm_id:<12} {true_val:<10} {str(predicted):<10} {match}")

    # 4. Accuracy
    correct  = sum(predictions[i] == removed_items[i] for i in removed_items)
    accuracy = correct / len(removed_items) * 100
    print(f"\nAccuracy: {correct}/{len(removed_items)} = {accuracy:.1f}%")

    # 5. Hierarchical update simulation
    #    Treat first 8 as reference, last 2 as "new arrivals"
    print("\n── Hierarchical Update ──────────────────────────────────────")
    adm_ids      = list(clinical_events.keys())
    reference_ds = {k: clinical_events[k] for k in adm_ids[:-2]}
    new_ds       = {k: masked_data[k]     for k in adm_ids[-2:]}

    knn_map, recomputed, frequency = hierarchical_update(
        new_sequences=new_ds,
        dataset=reference_ds,
        vocab=vocab,
        K=K,
        recompute_threshold=2
    )
    print(f"\n  Sequences flagged for neighbor recomputation: {recomputed}")

    return predictions, removed_items




# ************************************************************************************************************************************
# ************************************************************************************************************************************

# ── Run ───────────────────────────────────────────────────────────────────────

clinical_events = load_dict_from_pickle( '../Data/admission_diagnoses.pkl')
predictions, removed_items = run_pipeline(clinical_events, K=3)


# ************************************************************************************************************************************
# ************************************************************************************************************************************




# ************************************************************************************************************************************
# ************************************************************************************************************************************


