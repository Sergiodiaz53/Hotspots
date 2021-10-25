#Packages
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Tensorflow & tools
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#SKlearn tools
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

#Tools
from dna2vec.multi_k_model import MultiKModel

k = 5

#######################################################################
#Data loading##########################################################
#######################################################################

hotspots = np.load("Data/hotspots/kmers/hotspots-5k-list.npy")
labels = np.load("Data/hotspots/kmers/labels_hotspots-5k-list.npy")

#[OPTIONAL] limit number of samples to speed up training
hotspots, labels = shuffle(hotspots, labels, random_state = 0)
hotspots = hotspots[0:round((len(hotspots)/5))]
labels = labels[0:round((len(labels)/5))]

print('Hotspots loaded, length:', hotspots.shape)
print('Labels loaded, shape: ', labels.shape)

#######################################################################
#DNA2Vec###############################################################
#######################################################################

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)
mk_model = mk_model.model(k)

pretrained_weights = mk_model.wv.syn0
vocab_size, embedding_dim = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

def word2idx(word):
  return mk_model.wv.vocab[word].index
def idx2word(idx):
  return mk_model.wv.index2word[idx]

#######################################################################
#SeqTokenizer##########################################################
#######################################################################

hotspots_sequences = []

for idx, sample in enumerate(hotspots):
    current_seq = []
    for idx2, token in enumerate(sample):
        try:
            model_token = word2idx(token)
            current_seq.append(model_token)
        except:
            current_seq.append("-1")

    #Padding to fixedsize
    for i in range(len(sample), 1500):
        current_seq.append("0")


    current_seq.append(current_seq)

hotspots = hotspots_sequences
del hotspots_sequences

#######################################################################
#Neural Network########################################################
#######################################################################

#Hyperparameters
epochs=50
learning_rate = 0.01

#Model Definition
def createModel(vocab_size, embedding_dim):
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size,
                      output_dim=embedding_dim,
                      weights=[pretrained_weights]))
  model.add(Dropout(0))
  model.add(LSTM(embedding_dim))
  model.add(Dense(2, activation='softmax'))
  return model

def createOptimizer(model):

  optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)

  model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics = ['accuracy'])
  return model

model = createModel(vocab_size, embedding_dim)
model = createOptimizer(model)
model.summary()

#######################################################################
#Training##############################################################
#######################################################################

hotspots = np.array(hotspots)
x_train, x_test, y_train, y_test = train_test_split(hotspots, labels, test_size=0.1, shuffle=True)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_true_max = y_test

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, shuffle=True, verbose=2)

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