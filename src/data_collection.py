#Packages
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio import Seq

#Tools
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

#Personal Packages
from collection.utils  import *
from classification.hyperparameters import *

# Load hotspots
# Only hotspots of up to length 1,500 bp were taken
# Those that were shorter than 1,500 bp were padded with N's

hotspots = list(SeqIO.parse("Data/fasta/" + "combined-max-1500-padded-REMOVED-BAD.fasta", "fasta"))

# These random sequences were generated from all parts of the human genome
# They were generated with the exact length profile of the hotspots
# i.e. hotspots and sequences have the same amount of padding, which is up to 1,500 bps

nohotspots = list(SeqIO.parse("Data/fasta/" + "sample-max-1500-padded-REMOVED-BAD.fasta", "fasta"))

#######################################################################
#kmerList##############################################################
#######################################################################

USE_REVERSE=True
chunk_size=500


if(USE_REVERSE):
    print("Generating reverse hotspot list...")

    reversed_hotspots = []
    for i in hotspots:
        reversed_hotspots.append(i.reverse_complement())
    hotspots += reversed_hotspots

    reversed_hotspots = []
    for i in nohotspots:
        reversed_hotspots.append(i.reverse_complement())
    nohotspots += reversed_hotspots


print("Generating hotspot list...")
hotspots_list = getKmersListInChunks(hotspots,K,chunk_size)
nohotspots_list = getKmersListInChunks(nohotspots,K,chunk_size)

print("Generating frequency vectors with polys...")
hotspots_freq_vectors_with_interesting_polys = computeFrequecencyVectors(hotspots, K, INTERESTING_POLYS)
nohotspots_freq_vectors_with_interesting_polys = computeFrequecencyVectors(nohotspots, K, INTERESTING_POLYS)

print("Generating labels list...")
labels_hotspots = np.zeros(len(hotspots_list))
labels_nohotspots = np.ones(len(nohotspots_list))

hotspots = np.array(hotspots_list + nohotspots_list)
freq_vectors_with_interesting_polys = np.concatenate((hotspots_freq_vectors_with_interesting_polys, nohotspots_freq_vectors_with_interesting_polys), axis=0)
labels = np.concatenate((labels_hotspots, labels_nohotspots), axis=0)

print("Hotspots lenght: ", len(hotspots))
print("FreqVectors lenght: ", len(freq_vectors_with_interesting_polys))
print("Labels lenght: ", len(labels))

print("Saving datasets...")

reversed = ""
if(USE_REVERSE):
    reversed = "_with_reversed"

np.save("Data/kmers/hotspots-"+str(K)+"k-list-"+str(chunk_size)+"chunk"+reversed,hotspots)
np.save("Data/kmers/freqvectors_hotspots-"+str(K)+"k-polys-"+str(chunk_size)+"chunk"+reversed,freq_vectors_with_interesting_polys)
np.save("Data/kmers/labels_hotspots-"+str(K)+"k-list-"+str(chunk_size)+"chunk"+reversed,labels)