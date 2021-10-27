#Packages
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Tensorflow & tools
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard

#SKlearn tools
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Tools
from dna2vec.multi_k_model import MultiKModel

k = 5

#######################################################################
#Data loading##########################################################
#######################################################################

hotspots = np.load("Data/kmers/hotspots-5k-list-300chunk.npy")
labels = np.load("Data/kmers/labels_hotspots-5k-list-300chunk.npy")

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
mk_model = mk_model.model(k)

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

#Hyperparameters
epochs=100
learning_rate = 0.001
batch_size = 256

#Model Definition
def createModel(vocab_size, embedding_dim):
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size,
                      output_dim=embedding_dim,
                      weights=[pretrained_weights]))
  model.add(Dropout(0))
  #model.add(LSTM(100, return_sequences=True))
  model.add(Bidirectional(LSTM(units=16, kernel_initializer="glorot_normal")))
  model.add(Dense(1, activation='sigmoid'))
  return model

def createOptimizer(model):

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

  model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics = ['accuracy'])
  return model

model = createModel(vocab_size, embedding_dim)
model = createOptimizer(model)
model.summary()

tensorboard = TensorBoard(
  log_dir='.\logs',
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

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, callbacks=[tensorboard])

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