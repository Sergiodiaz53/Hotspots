import numpy as np
import tensorflow 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

root_dir = ""

hotspots = np.load(root_dir + "Data/kmers/hotspots-3k-list-500chunk.npy")
labels = np.load(root_dir + "Data/kmers/labels_hotspots-3k-list-500chunk.npy")

def generateVocabulary(dataset):
    vocab = set()
    for kmer_list in dataset:
        for kmer in kmer_list:
            vocab.add(kmer)
    return vocab

vocabulary = generateVocabulary(hotspots)
print(len(vocabulary))

vocab_size = len(vocabulary)
oov_token = 'oov'

hotspots = hotspots.tolist()
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(hotspots)
word_index = tokenizer.word_index

hotspots = tokenizer.texts_to_sequences(hotspots)

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

word_model = Word2Vec(hotspots, vector_size=100, epochs=20, compute_loss=True, callbacks=[callback()])

pretrained_weights = word_model.wv.syn0
vocab_size, embedding_dim = pretrained_weights.shape

word_model.save("kmer2vec.model")