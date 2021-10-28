
#Packages
import tensorflow as tf

#Tensorflow & tools
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def createBidirectionalLSTMModel(vocab_size, embedding_dim, pretrained_weights_for_embedding):
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size,
                      output_dim=embedding_dim,
                      weights=[pretrained_weights_for_embedding]))
  model.add(Dropout(0))
  model.add(Bidirectional(LSTM(units=16, kernel_initializer="glorot_normal", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
  model.add(Bidirectional(LSTM(units=8, kernel_initializer="glorot_normal", dropout=0.2, recurrent_dropout=0.2)))
  model.add(Dense(1, activation='sigmoid'))
  return model

def createOptimizer(model, learning_rate):

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

  model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics = ['accuracy'])
  return model