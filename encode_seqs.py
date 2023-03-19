import numpy as np
import os


def generate_sequences(n=10, length=5):
    return ["".join(np.random.choice(list("ACTG"), length)) for _ in range(n)]


def to_numpy(seqs):
    nucl_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    return np.vstack([
        np.array([nucl_map[ch] for ch in s], dtype=np.uint8) for s in seqs
    ])


def one_hot_encode(seq_array):
    n, length = seq_array.shape
    enc = np.vstack([np.eye(4), [0, 0, 0, 0]]).astype(np.uint8)
    return np.stack([enc[seq] for seq in seq_array], axis=0)


if __name__ == "__main__":

    # Two sequences
    seqs = ["ACGTTT", "GANNTT"]
    seq_array = to_numpy(seqs)
    seq_array_encoded = one_hot_encode(seq_array)
    print(seq_array_encoded.shape)
    print(seq_array_encoded)

    # Now let's process a bunch of randomly generated sequences
    seqs = generate_sequences(n=100_000, length=500)
    seq_array = to_numpy(seqs)
    seq_array_encoded = one_hot_encode(seq_array)
    np.save("encoded.npy", seq_array_encoded)
    fsize_mb = os.path.getsize('encoded.npy') / 1024 / 1204
    print(f"file size: {fsize_mb:.2f} MB")
    os.unlink("encoded.npy")  # remove it again

    # check that sum across last dimension is 1 (there should be exactly one 1
    # in each last dimension, if no nucleotide is N)
    assert np.all(np.sum(seq_array_encoded, axis=-1) == 1)

    # Ns in sequences should be encoded as [0,0,0,0,0]
    seq_array = to_numpy([list("NNNN")])
    seq_array_encoded = one_hot_encode(seq_array)
    assert np.all(seq_array_encoded == 0)
