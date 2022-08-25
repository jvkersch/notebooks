from argparse import ArgumentParser
from collections import Counter
from itertools import product
from csv import DictWriter
from Bio import SeqIO

def _make_triad_map():
    compact = {
        "AGV": 1,
        "ILFP": 2,
        "YMTS": 3,
        "HNQW": 4,
        "RK": 5,
        "DE": 6,
        "C": 7,
    }
    return {amino: code for (aminos, code) in compact.items() for amino in aminos}
    
TRIAD_MAP = _make_triad_map()
    
def get_triads(seq):
    for i in range(len(seq) - 2):
        yield seq[i:i+3]

def encode_triad(t):
    return tuple(TRIAD_MAP[a] for a in t)

def encode_sequence(seq):
    return Counter(encode_triad(t) for t in get_triads(seq))

def triad_name(t):
    return f"VS{t[0]}{t[1]}{t[2]}"

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("fasta", help="Protein sequence input FASTA file")
    ns = p.parse_args()
    
    triads = list(product(*((range(1, 8),)*3)))
    with open("conjoint_triad.csv", "w") as fp:
        writer = DictWriter(fp, fieldnames=["ID"] + [triad_name(t) for t in triads])
        writer.writeheader()
        
        for rec in SeqIO.parse(ns.fasta, "fasta"):
            data = {"ID": rec.name}
            counts = encode_sequence(rec.seq)
            for t in triads:
                data[triad_name(t)] = counts[t]

            writer.writerow(data)
