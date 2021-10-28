
#Packages
import tensorflow as tf

#Tensorflow & tools
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Activation, Add, concatenate

#Local
from hyperparameters import *

def createBidirectionalLSTMModel(vocab_size, embedding_dim, pretrained_weights_for_embedding):

    model_input = Input(shape=(vocab_size))
    output = Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[pretrained_weights_for_embedding])(model_input)
    
    output = Dropout(DROPOUT_RATE)(output)
    output = Bidirectional(LSTM(units=HIDDEN_UNITS_LSTM, kernel_initializer="glorot_normal", dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE, return_sequences=True))(output)
    output = Bidirectional(LSTM(units=int(HIDDEN_UNITS_LSTM/2), kernel_initializer="glorot_normal", dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=model_input, outputs=output)

    return model

def createBidirectionalLSTMModel_with_residual(vocab_size, embedding_dim, layers, freq_vector_size, pretrained_weights_for_embedding):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    lstm_input = Input(shape=(vocab_size))
    lstm_part  = Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[pretrained_weights_for_embedding])(lstm_input)

    lstm_part  = Dropout(DROPOUT_RATE)(lstm_part)
    lstm_part  = Bidirectional(LSTM(units=HIDDEN_UNITS_LSTM, kernel_initializer="glorot_normal", dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE, return_sequences=True))(lstm_part)
    lstm_part  = Bidirectional(LSTM(units=int(HIDDEN_UNITS_LSTM/2), kernel_initializer="glorot_normal", dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(lstm_part)


    #Input frequencyVectors+Polys
    res_input = Input(shape=(freq_vector_size))
    res_part = Dropout(DROPOUT_RATE)(res_input)

    for i in range(0, layers):

        def regression_identity_block(res_part, units_start, units_end, activation):
            res_shortcut = res_part

            ri_block = Dense(units = units_start, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_part)
            #output  = BatchNormalization()(output)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(units = units_start, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)
            #output  = BatchNormalization()(output)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(units = units_end, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)
            #output  = BatchNormalization()(output)

            ri_jump   = tf.keras.layers.Dense(units = units_end, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_shortcut)

            ri_block = Add()([ri_block, ri_jump])
            ri_block = Activation(activation=activation)(ri_block)
            return ri_block

        res_part = regression_identity_block(res_part, RESIDUAL_UNITS[i], RESIDUAL_UNITS[i+1], RESIDUAL_ACTIVATION_TYPE)

    
    output = concatenate([lstm_part, res_part], name = 'merge_parts')

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[lstm_input, res_input], outputs=output)

    
    return model

def createOptimizer(model, learning_rate):

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

  model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics = ['accuracy'])
  return model