
#Data options
K = 3

#Model selection
MODEL_NAMES = {
    0: 'bidirectionalLSTM',
    1: 'bidirectionalLSTM_with_kmer_frequency_vector'
}
MODEL_SELECTION = MODEL_NAMES[0]

#NN hyperparameters
EPOCHS=100
LEARNING_RATE = 0.001
BATCH_SIZE = 256

