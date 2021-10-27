#Tools
from dna2vec.multi_k_model import MultiKModel

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)
mk_model = mk_model.model(5)

pretrained_weights = mk_model.vectors
vocab_size, embedding_dim = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

def word2idx(word):
    return mk_model.key_to_index[word]
def idx2word(idx):
  return mk_model.wv.index_to_key[idx]

print(word2idx('ACG'))