import numpy as np
import pandas as pd
from Bio import SeqIO

def getOverlappingKmers(sequence, k=5):
    return [str(sequence[x:x+k]) for x in range(len(sequence) - k + 1)]

def getNonOverlappingKmers(sequence, k=5):
    return [str(sequence[x:(x+k)]) for x in range(0, len(sequence)-(len(sequence) % k), k)]

def getKmersListInChunks(dataset,k,chunk_size):
    dataset_list = []
    for i, seq_record in enumerate(dataset):
        kmer_list = getNonOverlappingKmers(seq_record.seq, k=k)
        for i in range(0, (len(kmer_list)-((len(kmer_list) % k))), chunk_size):
            dataset_list.append(kmer_list[i:i+chunk_size])

    return dataset_list
