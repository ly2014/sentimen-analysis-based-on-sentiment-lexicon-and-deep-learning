import numpy as np
import jieba
import collections
import re
# from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional, GRU, Dropout, Input, Convolution1D, MaxPool1D, Flatten,\
    concatenate, Conv1D
from keras import backend as K
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from lxml import etree


max_features = 27000
most_frequence = 1000
max_len = 300
batch_size = 32
vocab_dim = 768


# bc = BertClient()


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
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        content = etree.HTML(content)
        rs_data = content.xpath('//review/text()')
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
    label = [1 for i in range(5000)]
    label.extend([0 for i in range(5000)])
    label = np.array(label)
    data = []
    data.extend(load_data('sample.positive.txt'))
    data.extend(load_data('sample.negative.txt'))
    sentences = []
    for dt in data:
        dt = dt.replace('\n', '')
        dt = list(jieba.cut(dt))
        sentence = []
        for word in dt:
            if word not in stopwords and '\u4e00' <= word <= '\u9fa5':
                sentence.append(word)
        sentences.append(sentence)
    frequency = collections.Counter()
    # maxLen = 0
    # sumLen = 0
    # sumSentence = 0
    for sentence in sentences:
        # sumSentence += 1
        # count = 0
        for word in sentence:
            frequency[word] += 1
        #     count += 1
        #     sumLen += 1
        # if count > maxLen:
        #     maxLen = count
    # print(maxLen)
    # print(sumLen)
    # print(sumSentence)
    # print(len(frequency.keys()))
    word2index = dict()
    for i, x in enumerate(frequency.most_common(max_features)):
        word2index[x[0]] = i + 1
    # word2vec = dict()
    # vecs = []
    # words = []
    # for i in word2index.keys():
    #     word2vec[i] = bc.encode([i])[0]
    #     words.append(i)
    #     vecs.append(word2vec[i])
    # with open('word.txt', 'w', encoding='utf-8') as f:
    #     for i in words:
    #         f.write(i + '\n')
    # vecs = np.array(vecs)
    # np.savetxt("vec.txt", vecs)
    #
    word2vec = dict()
    vecs = np.loadtxt('vec.txt')
    words = []
    with open('word.txt', 'r', encoding='utf-8') as f:
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
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Conv1D(256, 3, padding='valid', activation='relu'))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', recall, fmeasure])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=[x_test, y_test])
    print(model.evaluate(x_test, y_test, batch_size=batch_size))

    # model = Sequential()
    # # 使用Embedding层将每个词编码转换为词向量
    # model.add(Embedding(max_features + 1,  # 表示文本数据中词汇的取值可能数,从语料库之中保留多少个单词。 因为Keras需要预留一个全零层， 所以+1
    #                     vocab_dim,  # 嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小
    #                     weights=[embedding_weights],
    #                     input_length=max_len,  # 输入序列的长度，也就是一次输入带有的词汇个数
    #                     trainable=False  # 我们设置 trainable = False，代表词向量不作为参数进行更新
    #                     ))
    # # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    # main_input = Input(shape=(max_len,), dtype='float64')
    # # 词嵌入（使用预训练的词向量）
    # embedder = Embedding(max_features + 1, vocab_dim, input_length=max_len, weights=[embedding_weights], trainable=False)
    # embed = embedder(main_input)
    # # 词窗大小分别为3,4,5
    # cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    # cnn1 = MaxPool1D(pool_size=4)(cnn1)
    # cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    # cnn2 = MaxPool1D(pool_size=4)(cnn2)
    # cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    # cnn3 = MaxPool1D(pool_size=4)(cnn3)
    # # 合并三个模型的输出向量
    # cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    # flat = Flatten()(cnn)
    # drop = Dropout(0.5)(flat)
    # main_output = Dense(1, activation='sigmoid')(drop)
    # model = Model(inputs=main_input, outputs=main_output)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy', recall, fmeasure])
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=5,
    #           validation_data=[x_test, y_test])
    # print(model.evaluate(x_test, y_test, batch_size=batch_size))
    # svc = SVC()
    # svc.fit(x_train, y_train)
    # y_pred = svc.predict(x_test)
    # bayes = MultinomialNB()
    # bayes.fit(x_train, y_train)
    # y_pred = bayes.predict(x_test)
    # print(metrics.accuracy_score(y_test, y_pred))
    # print(metrics.recall_score(y_test, y_pred))
    # print(metrics.f1_score(y_test, y_pred))
