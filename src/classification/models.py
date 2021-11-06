
#Packages
import tensorflow as tf

#Tensorflow & tools
from tensorflow import keras
from keras import Model, Sequential, regularizers
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Activation, Add, concatenate

from .hyperparameters import *

def createModel(vocab_size, embedding_dim, pretrained_weights_for_embedding):
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size,
                      output_dim=embedding_dim,
                      weights=[pretrained_weights_for_embedding]))
  model.add(LSTM(HIDDEN_UNITS_LSTM))
  model.add(Dense(1, activation='sigmoid'))
  return model

def createResNetModel(freq_vector_size, layers):
    initializer = keras.initializers.GlorotNormal()

    def regression_identity_block(res_part, activation,):
          res_shortcut = res_part

          ri_block = Dense(units = 32 , kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_part)
          #ri_block  = BatchNormalization()(ri_block)
          ri_block = Activation(activation=activation)(ri_block)

          ri_block = Dense(units = 16, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)
          #ri_block  = BatchNormalization()(ri_block)
          ri_block = Activation(activation=activation)(ri_block)

          ri_block = Dense(8, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)

          ri_jump   = Dense(8, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_shortcut)

          ri_block = Add()([ri_block, ri_jump])
          #ri_block  = BatchNormalization()(ri_block)
          ri_block = Activation(activation=activation)(ri_block)
          return ri_block

    model_input = Input(shape=(freq_vector_size))
    res_part = Dropout(DROPOUT_RATE)(model_input)

    for i in range(0, layers):
        res_part = regression_identity_block(res_part, RESIDUAL_ACTIVATION_TYPE)

    output = Dense(1, activation='sigmoid')(res_part)

    model = Model(inputs=model_input, outputs=output)
    model = createOptimizer(model, LEARNING_RATE)

    return model, model_input, output

def createBidirectionalLSTMModel(seq_lenght, vocab_size, embedding_dim, pretrained_weights_for_embedding):
    initializer = keras.initializers.GlorotNormal()

    model_input = Input(shape=(seq_lenght))
    output = Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=seq_lenght,
                        weights=[pretrained_weights_for_embedding])(model_input)
    
    output  = Bidirectional(LSTM(units=HIDDEN_UNITS_LSTM, kernel_initializer=initializer,
                                     kernel_regularizer=regularizers.l2(L2_RATE),
                                    dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(output)    
    #output = Bidirectional(LSTM(units=int(HIDDEN_UNITS_LSTM/2), kernel_initializer="glorot_normal", dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=model_input, outputs=output)

    return model, model_input, output

def createBidirectionalLSTMModel_with_residual(seq_lenght, vocab_size, embedding_dim, layers, freq_vector_size, pretrained_weights_for_embedding):
    initializer = keras.initializers.GlorotNormal()

    lstm_input = Input(shape=(seq_lenght))
    lstm_part  = Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=seq_lenght,
                        weights=[pretrained_weights_for_embedding])(lstm_input)

    lstm_part  = Bidirectional(LSTM(units=HIDDEN_UNITS_LSTM, kernel_initializer=initializer,
                                     kernel_regularizer=regularizers.l2(L2_RATE),
                                    dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(lstm_part)
    #lstm_part  = Bidirectional(LSTM(units=int(HIDDEN_UNITS_LSTM/2), kernel_initializer=initializer, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(lstm_part)

    lstm_part = Dense(units = HIDDEN_UNITS_DENSE_LSTM, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(lstm_part)
    lstm_part = BatchNormalization()(lstm_part)
    lstm_part = Activation(activation='relu')(lstm_part)

    #Input frequencyVectors+Polys
    res_input = Input(shape=(freq_vector_size))
    res_part = Dropout(DROPOUT_RATE)(res_input)

    for i in range(0, layers):

        def regression_identity_block(res_part, activation):
            res_shortcut = res_part

            ri_block = Dense(units = RESIDUAL_UNITS[0] , kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_part)
            ri_block  = BatchNormalization()(ri_block)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(units = RESIDUAL_UNITS[1], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)
            ri_block  = BatchNormalization()(ri_block)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(RESIDUAL_UNITS[2], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)

            ri_jump   = Dense(RESIDUAL_UNITS[2], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_shortcut)

            ri_block = Add()([ri_block, ri_jump])
            ri_block  = BatchNormalization()(ri_block)
            ri_block = Activation(activation=activation)(ri_block)
            return ri_block

        res_part = regression_identity_block(res_part, RESIDUAL_ACTIVATION_TYPE)

    
    output = concatenate([lstm_part, res_part], name = 'merge_parts')

    output = Dense(units = 4, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(output)
    output = BatchNormalization()(output)
    output = Activation(activation='relu')(output)

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[lstm_input, res_input], outputs=output)
    
    return model

def createBidirectionalLSTMModel_with_residual_without_batch_normalization(seq_lenght, vocab_size, embedding_dim, layers, freq_vector_size, pretrained_weights_for_embedding):
    initializer = keras.initializers.GlorotNormal()

    lstm_input = Input(shape=(seq_lenght))
    lstm_part  = Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=seq_lenght,
                        weights=[pretrained_weights_for_embedding])(lstm_input)

    lstm_part  = Bidirectional(LSTM(units=HIDDEN_UNITS_LSTM, kernel_initializer=initializer,
                                     kernel_regularizer=regularizers.l2(L2_RATE),
                                    dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))(lstm_part)

    lstm_part = Dense(units = HIDDEN_UNITS_DENSE_LSTM, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(lstm_part)
    lstm_part = Activation(activation='relu')(lstm_part)

    #Input frequencyVectors+Polys
    res_input = Input(shape=(freq_vector_size))
    res_part = Dropout(DROPOUT_RATE)(res_input)

    for i in range(0, layers):

        def regression_identity_block(res_part, activation):
            res_shortcut = res_part

            ri_block = Dense(units = RESIDUAL_UNITS[0] , kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_part)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(units = RESIDUAL_UNITS[1], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)
            ri_block = Activation(activation=activation)(ri_block)

            ri_block = Dense(RESIDUAL_UNITS[2], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(ri_block)

            ri_jump   = Dense(RESIDUAL_UNITS[2], kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(res_shortcut)

            ri_block = Add()([ri_block, ri_jump])
            ri_block = Activation(activation=activation)(ri_block)
            return ri_block

        res_part = regression_identity_block(res_part, RESIDUAL_ACTIVATION_TYPE)

    
    output = concatenate([lstm_part, res_part], name = 'merge_parts')

    output = Dense(units = 4, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(output)
    output = Activation(activation='relu')(output)

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[lstm_input, res_input], outputs=output)
    
    return model

def createPreTrainedResLSTMModel_with_residual(seq_lenght, vocab_size, embedding_dim, layers, freq_vector_size, pretrained_weights_for_embedding, root_dir):
    initializer = keras.initializers.GlorotNormal()

    #Load pretrained ResNet
    resnet, res_inputs, res_outputs = createResNetModel(freq_vector_size=542, layers = RESIDUAL_LAYERS)
    pretrained_resnet = keras.models.load_model(root_dir+'pretrained_models/ResNetmodel.h5')
    resnet.set_weights(pretrained_resnet.get_weights())
    for layer in resnet.layers:
        layer.trainable = False
    res_outputs = resnet.layers[-2].output

    #Load pretrained LSTM
    lstmnet, lstm_inputs, lstm_outputs = createBidirectionalLSTMModel(seq_lenght, vocab_size, embedding_dim, pretrained_weights_for_embedding)
    pretrained_lstmnet = keras.models.load_model(root_dir+'pretrained_models/LSTM_model.h5')
    lstmnet.set_weights(pretrained_lstmnet.get_weights())
    for layer in lstmnet.layers:
        layer.trainable = False
    lstm_outputs = lstmnet.layers[-2].output

    output = concatenate([res_outputs,lstm_outputs], name = 'merge_parts')

    output = Dense(units = 4, kernel_initializer=initializer, use_bias=True, bias_initializer='zeros')(output)
    output = Activation(activation='relu')(output)

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[lstm_inputs, res_inputs], outputs=output)

    return model

def createOptimizer(model, learning_rate):

  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics = ['accuracy'])
  return model