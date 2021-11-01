
#Data options
K = 3
INTERESTING_POLYS = ['AAAAAAAAAAAA', 'TTTTTTTTTTTT', 'TGTGTGTGTGTG', 'GTGTGTGTGTGT', 'CACACACACACA', 'ACACACACACAC',
                     'ATATATATATAT', 'TATATATATATA', 'TTAAAAAAAAAA', 'TTTTTTTTTTAA', 'CTGTAATCCCAG', 'CCTGTAATCCCA',
                     'CTGGGATTACAG', 'TGGGATTACAGG', 'TGTAATCCCAGC', 'CCTCAGCCTCCC', 'GCTGGGATTACA', 'GGGAGGCTGAGG',
                     'CCTTTTTTTTTT', 'AAAAAAAAAAGG', 'AAAAAAAGAAAG', 'CTTTCTTTTTTT', 'TAAAAATAAAAA', 'TTTTTATTTTTA',
                     'CCAAAAAAAAAA', 'GCCTCAGCCTCC', 'TTTTTTTTTTGG', 'CTTTTTTTTTTG', 'CAAAAAAAAAAG', 'GGAGGCTGAGGC' ]

#Model selection
MODEL_NAMES = {
    0: 'basicTestModel',
    1: 'bidirectionalLSTM',
    2: 'bidirectionalLSTM_with_residual',
    3: 'bidirectionalLSTM_with_residual_without_batch_normalization'
}
MODEL_SELECTION = MODEL_NAMES[3]

# NN hyperparameters
EPOCHS = 200
LEARNING_RATE = 0.01
BATCH_SIZE = 256
DROPOUT_RATE = 0.3

# LSTM
HIDDEN_UNITS_LSTM = 8
HIDDEN_UNITS_DENSE_LSTM = 8
RECURRENT_DROPOUT_RATE = 0.2
L2_RATE = 1e-05

# Residual
RESIDUAL_LAYERS = 1
RESIDUAL_UNITS = [16,8,4]
RESIDUAL_ACTIVATION_TYPE = 'relu'
