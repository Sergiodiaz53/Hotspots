#Packages
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#SKlearn tools
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Tools
from dna2vec.multi_k_model import MultiKModel
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

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

reversed = ""
if(USE_REVERSE):
    reversed = "_with_reversed"

print("Loading hotspots...")
hs_train = np.load(root_dir+"Data/final/hs_train-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
hs_test = np.load(root_dir+"Data/final/hs_test-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
print('Hotspots loaded, train:', hs_train.shape, ", test: ", hs_test.shape)

print("Loading frequency vectors...")
fv_train = np.load(root_dir+"Data/final/fv_train-"+str(K_FV)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
fv_test = np.load(root_dir+"Data/final/fv_test-"+str(K_FV)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
print('Frequency vector and polys loaded, train:', fv_train.shape, ", test: ", fv_test.shape)

print("Loading labels...")
y_train = np.load(root_dir+"Data/final/y_train-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
y_test = np.load(root_dir+"Data/final/y_test-"+str(K)+"k-list-"+str(CHUNK_SIZE)+"chunk"+reversed+".npy")
print('Labels loaded, train:', y_train.shape, ", test: ", y_test.shape)

#[OPTIONAL] limit number of samples to speed up training
if (LIMIT_TRAINING_SIZE > 1) :
  size_train = round(len(hs_train)/LIMIT_TRAINING_SIZE)
  hs_train = hs_train[0:size_train]
  fv_train = fv_train[0:size_train]
  y_train = y_train[0:size_train]

hs_train = hs_train.astype('float32')
hs_test = hs_test.astype('float32')

fv_train = fv_train.astype('float32')
fv_test = fv_test.astype('float32')
seq_size = len(hs_train[0])

#######################################################################
#DNA2Vec###############################################################
#######################################################################

filepath = root_dir+'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)
mk_model = mk_model.model(K)

pretrained_weights = mk_model.vectors
vocab_size, embedding_dim = pretrained_weights.shape

def word2idx(word):
    return mk_model.key_to_index[word]
def idx2word(idx):
  return mk_model.wv.index_to_key[idx]

#######################################################################
#Neural Network########################################################
#######################################################################

#Define Callbacks
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=100, min_delta=0.001, cooldown=50, min_lr=0.0001)
tensorboard = TensorBoard(log_dir= run_dir+'logs/', histogram_freq=1, write_images=True, write_graph=True) 
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=run_dir+"checkpoint", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

#Define model
print("Generating model...")
if(MODEL_SELECTION=='bidirectionalLSTM'):
  model, _, _ = createBidirectionalLSTMModel(seq_size, vocab_size, embedding_dim, pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual'):
  model = createBidirectionalLSTMModel_with_residual(seq_size, vocab_size, embedding_dim, freq_vector_size=len(fv_train[0]), layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual_without_batch_normalization'):
  model = createBidirectionalLSTMModel_with_residual_without_batch_normalization(seq_size, vocab_size, embedding_dim, freq_vector_size=len(fv_train[0]), layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual_without_batch_normalization'):
  model = createBidirectionalLSTMModel_with_residual_without_batch_normalization(seq_size, vocab_size, embedding_dim, freq_vector_size=len(fv_train[0]), layers=RESIDUAL_LAYERS, pretrained_weights_for_embedding=pretrained_weights)
elif(MODEL_SELECTION=='Pretrained_bidirectionalLSTM_with_residual_without_batch_normalization'):
  model = createPreTrainedResLSTMModel_with_residual(seq_size, vocab_size, embedding_dim, layers=RESIDUAL_LAYERS, freq_vector_size=len(fv_train[0]), pretrained_weights_for_embedding=pretrained_weights, root_dir=root_dir)
elif(MODEL_SELECTION=='basicTestModel'):
  model = createModel(vocab_size, embedding_dim, pretrained_weights)

keras.utils.plot_model(model, run_dir+'multi_input_and_output_model.png', show_shapes=True)
  
model = createOptimizer(model, LEARNING_RATE)
model.summary()

#######################################################################
#Training##############################################################
#######################################################################
#TODO: Create list with inputs and validations to avoid IFs

if(MODEL_SELECTION=='bidirectionalLSTM'):
  history = model.fit(hs_train, y_train, validation_data=(hs_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, model_checkpoint])
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual'):
  history = model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping, model_checkpoint])
elif(MODEL_SELECTION=='bidirectionalLSTM_with_residual_without_batch_normalization'):
  history = model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping, model_checkpoint])
elif(MODEL_SELECTION=='Pretrained_bidirectionalLSTM_with_residual_without_batch_normalization'):
  history = model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, model_checkpoint])
elif(MODEL_SELECTION=='basicTestModel'):
  history = model.fit(hs_train, y_train, validation_data=(hs_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, early_stopping, model_checkpoint])

#######################################################################
#Results###############################################################
#######################################################################

print("Loading best model...")
model.load_weights(run_dir+"checkpoint")

#TODO: Create output list to implement results
"""y_pred=model.predict(hs_test)
class_names = ["Hotspot", "No Hotspot"]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index = class_names, columns = class_names)

print('Accuracy Y_test: ', accuracy_score(y_test, y_pred))
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(run_dir+'confussion_matrix.png')"""

print("Saving best model...")
model.save(run_dir+'model.h5')
