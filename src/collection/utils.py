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

# Function to find different poly's in a hotspot and add them as features
def computePolys(string, INTERESTING_POLYS):
    matched_polys = np.zeros(len(INTERESTING_POLYS))
    for idx, poly in enumerate(INTERESTING_POLYS):
        #[NOT IN USE] Sum of found polys
        """
        res = len(re.findall(poly, string))
        matched_polys[idx] = res
        """
        #Polys at 1 if found
        found = string.find(poly)
        if (found != -1):
            matched_polys[idx] = 1
        
    return matched_polys

def computeHash(string):
    hashv = 0
    value = {"A":0, "C":1, "G":2, "T":3}
    i = len(string)-1
    for nucl in string:
        if(nucl == 'N'): return -1
        hashv = hashv + (4**i) * value[nucl]
        i = i - 1
    return hashv

def computeFrequecencyVectors(sequences_dataset, K, INTERESTING_POLYS):
    nmers = 4**K
    hotspots_vector = np.zeros((len(sequences_dataset)*(4**K + len(INTERESTING_POLYS)))).reshape(len(sequences_dataset), (4**K + len(INTERESTING_POLYS)))
    for i, seq_record in enumerate(sequences_dataset):
        for kmer in getOverlappingKmers(seq_record.seq, k=K):
            hashv = computeHash(kmer)
            if(hashv > -1): hotspots_vector[i, hashv] = hotspots_vector[i, hashv] + 1
        hotspots_vector[i, nmers:] = computePolys(str(seq_record.seq), INTERESTING_POLYS)
    return hotspots_vector
