#Packages
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO

#Tools
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#Personal Packages
from collection.utils  import *
from hyperparameters import *

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

chunk_size=500

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

np.save("Data/kmers/hotspots-"+str(K)+"k-list-"+str(chunk_size)+"chunk",hotspots)
np.save("Data/kmers/freqvectors_hotspots-"+str(K)+"k-polys-"+str(chunk_size)+"chunk",freq_vectors_with_interesting_polys)
np.save("Data/kmers/labels_hotspots-"+str(K)+"k-list-"+str(chunk_size)+"chunk",labels)