#Packages
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#SKlearn tools
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Tools
from dna2vec.multi_k_model import MultiKModel
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

#Local
from classification.models import *
from classification.hyperparameters import *

######################################################################
#Data loading##########################################################
#######################################################################


try:
  run_dir = str(sys.argv[1])
  root_dir = "../"
except:
  run_dir = ""
  root_dir = ""

hotspots = np.load(root_dir+"Data/kmers/hotspots-3k-list-500chunk_with_reversed.npy")
freq_vectors = np.load(root_dir+"Data/kmers/freqvectors_hotspots-3k-polys-500chunk_with_reversed.npy")
labels = np.load(root_dir+"Data/kmers/labels_hotspots-3k-list-500chunk_with_reversed.npy")

print('Hotspots loaded, shape:', hotspots.shape)
print('Frequency vector and polys loaded, shape:', freq_vectors.shape)
print('Labels loaded, shape: ', labels.shape)

#[OPTIONAL] limit number of samples to speed up training
hotspots, freq_vectors, labels = shuffle(hotspots, freq_vectors, labels, random_state = 0)
hotspots = hotspots[0:round((len(hotspots)))]
freq_vectors = freq_vectors[0:round((len(freq_vectors)))]
labels = labels[0:round((len(labels)))]


#######################################################################
#DNA2Vec###############################################################
#######################################################################

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
#Neural Network########################################################
#######################################################################

if(MODEL_SELECTION=='bidirectionalLSTM'):
  model = createBidirectionalLSTMModel(seq_size, vocab_size, embedding_dim, pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual'):
  model = createBidirectionalLSTMModel_with_residual(seq_size, vocab_size, embedding_dim, freq_vector_size=len(freq_vectors[0]), layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual_without_batch_normalization'):
  model = createBidirectionalLSTMModel_with_residual_without_batch_normalization(seq_size, vocab_size, embedding_dim, freq_vector_size=len(freq_vectors[0]), layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)
elif(MODEL_SELECTION=='basicTestModel'):
  model = createModel(vocab_size, embedding_dim, pretrained_weights)

keras.utils.plot_model(model, run_dir+'multi_input_and_output_model.png', show_shapes=True)
  
model = createOptimizer(model, LEARNING_RATE)
model.summary()

#Define Callbacks

reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_delta=0.001, cooldown=20, min_lr=0.0001)

tensorboard = TensorBoard(log_dir= run_dir+'logs/', histogram_freq=1, write_images=True, write_graph=True) 

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=60, restore_best_weights=True)

#######################################################################
#Training##############################################################
#######################################################################

hotspots = np.array(hotspots)
hs_train, hs_test, fv_train, fv_test, y_train, y_test = train_test_split(hotspots, freq_vectors, labels, test_size=0.2, shuffle=True)

hs_train = hs_train.astype('float32')
hs_test = hs_test.astype('float32')

fv_train = fv_train.astype('float32')
fv_test = fv_test.astype('float32')

y_true_max = y_test

if(MODEL_SELECTION=='bidirectionalLSTM'):
  history = model.fit(hs_train, y_train, validation_data=(hs_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping])
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual'):
  history = model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping])
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual_without_batch_normalization'):
  history = model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping])
elif(MODEL_SELECTION=='basicTestModel'):
  history = model.fit(hs_train, y_train, validation_data=(hs_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping])

#######################################################################
#Results###############################################################
#######################################################################

y_pred=np.argmax(model.predict([hs_test,fv_test]), axis=-1)
class_names = ["Hotspot", "No Hotspot"]
con_mat = tf.math.confusion_matrix(labels=y_true_max, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index = class_names, columns = class_names)

print('Accuracy Y_test: ', accuracy_score(y_true_max, y_pred))
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(run_dir+'confussion_matrix.png')

model.save(run_dir+'model.h5')
