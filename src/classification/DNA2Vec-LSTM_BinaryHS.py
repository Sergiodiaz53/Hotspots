#Packages

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
from tensorflow.keras.callbacks import TensorBoard

#Local
from hyperparameters import *
from models import *

#######################################################################
#Data loading##########################################################
#######################################################################

hotspots = np.load("Data/kmers/hotspots-3k-list-500chunk.npy")
labels = np.load("Data/kmers/labels_hotspots-3k-list-500chunk.npy")

#[OPTIONAL] limit number of samples to speed up training
hotspots, labels = shuffle(hotspots, labels, random_state = 0)
hotspots = hotspots[0:round((len(hotspots)))]
labels = labels[0:round((len(labels)))]

print('Hotspots loaded, length:', hotspots.shape)
print('Labels loaded, shape: ', labels.shape)

#######################################################################
#DNA2Vec###############################################################
#######################################################################

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
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

#######################################################################
#Neural Network########################################################
#######################################################################

if(MODEL_SELECTION=='bidirectionalLSTM'):
  model = createBidirectionalLSTMModel(vocab_size, embedding_dim, pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual'):
  model = createBidirectionalLSTMModel_with_residual(vocab_size, embedding_dim, freq_vector_size=512, layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)

keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
  
model = createOptimizer(model, LEARNING_RATE)
model.summary()

tensorboard = TensorBoard(
  log_dir='logs/',
  histogram_freq=1,
  write_images=True
) 

#######################################################################
#Training##############################################################
#######################################################################

hotspots = np.array(hotspots)
x_train, x_test, y_train, y_test = train_test_split(hotspots, labels, test_size=0.2, shuffle=True)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_true_max = y_test

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard])

#######################################################################
#Results###############################################################
#######################################################################

y_pred=np.argmax(model.predict(x_test), axis=-1)
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
plt.savefig('confussion_matrix.png')