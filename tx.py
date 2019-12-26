import numpy as np
import jieba
import collections
import re
# from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional, GRU, Dropout, Input, Convolution1D, MaxPool1D, Flatten,\
    concatenate, Conv1D, TimeDistributed
from keras import backend as K
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


max_features = 50000
most_frequence = 1000
max_len = 12
batch_size = 32
vocab_dim = 768


# bc = BertClient()

class Attention(Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


def load_data(filename):
    rs_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('[^\u4e00-\u9fa5]', '', line)
            rs_data.append(line)
    return rs_data


def get_vec(senti_words, word, vec):
    if word in senti_words.keys():
        return vec * senti_words[word]
    return vec


if __name__ == '__main__':
    stopwords = []
    sentiments = []
    sentivalue = {}
    with open('stop.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    with open('sentiment.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            sentiments.append(line[0])
            sentivalue[line[0]] = float(line[1])
    jieba.load_userdict(sentiments)
    label = [1 for j in range(50000)]
    label.extend([0 for j in range(50000)])
    label = np.array(label)
    data = []
    data.extend(load_data('p.txt'))
    data.extend(load_data('n.txt'))
    sentences = []
    for dt in data:
        dt = list(jieba.cut(dt))
        sentence = []
        for word in dt:
            if word not in stopwords:
                sentence.append(word)
        sentences.append(sentence)
    frequency = collections.Counter()
    maxLen = 0
    sumLen = 0
    sumSentence = 0
    for sentence in sentences:
        sumSentence += 1
        count = 0
        for word in sentence:
            frequency[word] += 1
            count += 1
            sumLen += 1
        if count > maxLen:
            maxLen = count
    file = open('rs.txt', 'w', encoding='utf-8')
    for i in range(8):
        max_features = 50000 - 5000 * i
        word2index = dict()
        for i, x in enumerate(frequency.most_common(max_features)):
            word2index[x[0]] = i + 1

        word2vec = dict()
        vecs = np.loadtxt('vec_t.txt')
        words = []
        with open('word_t.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                words.append(line.strip())
        for i in range(len(words)):
            # word2vec[words[i]] = vecs[i]
            word2vec[words[i]] = get_vec(sentivalue, words[i], vecs[i])

        new_sentences = []
        for sentence in sentences:
            sen = []
            for word in sentence:
                if word in word2index.keys():
                    sen.append(word2index[word])
                else:
                    sen.append(0)
            new_sentences.append(sen)
        new_sentences = np.array(new_sentences)
        x_train, x_test, y_train, y_test = train_test_split(new_sentences, label, test_size=0.2)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        embedding_weights = np.zeros((max_features + 1, vocab_dim))
        for x, i in word2index.items():
            embedding_weights[i, :] = word2vec[x]
        model = Sequential()
        model.add(Embedding(output_dim=vocab_dim, input_dim=max_features + 1, weights=[embedding_weights],
                            input_length=max_len))
        model.add(Conv1D(128, 3, padding='valid'))
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Attention(10))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', recall, fmeasure])
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=5,
                  validation_data=[x_test, y_test])
        file.write(str(max_features) + ': ')
        file.write(str(model.evaluate(x_test, y_test, batch_size=batch_size)))
        file.write('\n')

