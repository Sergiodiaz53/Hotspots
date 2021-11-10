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

#Define Callbacks
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=100, min_delta=0.001, cooldown=0, min_lr=1e-8)
tensorboard = TensorBoard(log_dir= run_dir+'logs/', histogram_freq=1, write_images=True, write_graph=True) 
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=run_dir+"checkpoint", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

model = keras.models.load_model(root_dir+'pretrained_models/ensemble_69515.h5')

for layer in model.layers:
    layer.trainable = True

keras.utils.plot_model(model, run_dir+'multi_input_and_output_model.png', show_shapes=True)
  
model = createOptimizer(model, 1e-6)
model.summary()

model.fit([hs_train, fv_train], y_train, validation_data=([hs_test,fv_test], y_test), epochs=1000, batch_size=64, shuffle=True, verbose=2, callbacks=[tensorboard, reduce_lr, model_checkpoint])

print("Loading best model...")
model.load_weights(run_dir+"checkpoint")

print("Saving best model...")
model.save(run_dir+'model.h5')