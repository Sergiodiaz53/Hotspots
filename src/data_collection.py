#Packages
import sys
import numpy as np
from Bio import SeqIO

#Tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dna2vec.multi_k_model import MultiKModel

#Personal Packages
from collection.utils  import *
from classification.hyperparameters import *

try:
  run_dir = str(sys.argv[1])
  root_dir = "../"
except:
  run_dir = ""
  root_dir = ""

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
hotspots_list = getKmersListInChunks(hotspots,K,CHUNK_SIZE)
nohotspots_list = getKmersListInChunks(nohotspots,K,CHUNK_SIZE)

print("Generating frequency vectors with polys...")
hotspots_freq_vectors_with_interesting_polys = computeFrequecencyVectors(hotspots, K_FV, INTERESTING_POLYS)
nohotspots_freq_vectors_with_interesting_polys = computeFrequecencyVectors(nohotspots, K_FV, INTERESTING_POLYS)

print("Generating labels list...")
labels_hotspots = np.zeros(len(hotspots_list))
labels_nohotspots = np.ones(len(nohotspots_list))

hotspots = np.array(hotspots_list + nohotspots_list)
freq_vectors_with_interesting_polys = np.concatenate((hotspots_freq_vectors_with_interesting_polys, nohotspots_freq_vectors_with_interesting_polys), axis=0)
if(K_FV == 5): freq_vectors_with_interesting_polys = np.delete(freq_vectors_with_interesting_polys, np.s_[512:1024], axis=1)
labels = np.concatenate((labels_hotspots, labels_nohotspots), axis=0)

print("Hotspots lenght: ", len(hotspots))
print("FreqVectors lenght: ", len(freq_vectors_with_interesting_polys))
print("Labels lenght: ", len(labels))

print("Saving datasets...")

reversed = ""
if(USE_REVERSE):
    reversed = "_with_reversed"

np.save("Data/kmers/hotspots-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,hotspots)
np.save("Data/kmers/freqvectors_hotspots-"+str(K)+"k-polys-"+str(CHUNK_SIZE)+"chunk"+reversed,freq_vectors_with_interesting_polys)
np.save("Data/kmers/labels_hotspots-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,labels)

#######################################################################
#DNA2Vec###############################################################
#######################################################################

print('Loading DNA2VEC...')

filepath = root_dir+'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)
mk_model = mk_model.model(K)

pretrained_weights = mk_model.vectors
vocab_size, embedding_dim = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

def word2idx(word):
    return mk_model.key_to_index[word]
def idx2word(idx):
  return mk_model.wv.index_to_key[idx]

#######################################################################
#SeqTokenizer##########################################################
#######################################################################

print('Tokenizing with DNA2Vec...')

hotspots_sequences = []

for idx, sample in enumerate(hotspots):
    current_seq = []
    for idx2, token in enumerate(sample):
      token = token.upper()
      try:
          model_token = word2idx(token)
          current_seq.append(model_token)
      except:
          current_seq.append("0")

    hotspots_sequences.append(current_seq)

hotspots = hotspots_sequences
seq_size = len(hotspots[0])

#######################################################################
#Normalizing FrequencyVectors##########################################
#######################################################################

scaler = MinMaxScaler()
freq_vectors_with_interesting_polys = scaler.fit_transform(freq_vectors_with_interesting_polys)

#######################################################################
#Neural Network########################################################
#######################################################################

print('Splitting datasets...')

hotspots = np.array(hotspots)
labels = np.array(labels)
hs_train, hs_test, fv_train, fv_test, y_train, y_test = train_test_split(hotspots, freq_vectors_with_interesting_polys, labels, test_size=0.2, shuffle=True)

hs_train = hs_train.astype('float32')
hs_test = hs_test.astype('float32')

fv_train = fv_train.astype('float32')
fv_test = fv_test.astype('float32')

print('Saving hotspots...')
np.save("Data/final/hs_train-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,hs_train)
np.save("Data/final/hs_test-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,hs_test)

print('Saving frequency vectors...')
np.save("Data/final/fv_train-"+str(K_FV)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,fv_train)
np.save("Data/final/fv_test-"+str(K_FV)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,fv_test)

print('Saving labels...')
np.save("Data/final/y_train-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,y_train)
np.save("Data/final/y_test-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed,y_test)
