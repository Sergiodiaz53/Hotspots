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
from utils import *

# Load hotspots
# Only hotspots of up to length 1,500 bp were taken
# Those that were shorter than 1,500 bp were padded with N's

hotspots = list(SeqIO.parse("Data/fasta/" + "combined-max-1500-padded-REMOVED-BAD.fasta", "fasta"))

# These random sequences were generated from all parts of the human genome
# They were generated with the exact length profile of the hotspots
# i.e. hotspots and sequences have the same amount of padding, which is up to 1,500 bps

nohotspots = list(SeqIO.parse("Data/fasta/" + "sample-max-1500-padded-REMOVED-BAD.fasta", "fasta"))


k=5
chunk_size = 300

print("Generating hotspot list...")
hotspots_list = getKmersListInChunks(hotspots,k,chunk_size)
del hotspots

print("Generating no-hotspot list...")
nohotspots_list = getKmersListInChunks(nohotspots,k,chunk_size)
del nohotspots

labels_hotspots = np.zeros(len(hotspots_list))
labels_nohotspots = np.ones(len(nohotspots_list))

hotspots = np.array(hotspots_list + nohotspots_list)
labels = np.concatenate((labels_hotspots, labels_nohotspots), axis=0)

print("Saving dataset...")

np.save("Data/kmers/hotspots-5k-list-"+str(chunk_size)+"chunk",hotspots)
np.save("Data/kmers/labels_hotspots-5k-list-"+str(chunk_size)+"chunk",labels)