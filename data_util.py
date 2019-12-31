import jieba
import numpy as np


def process_sentiment_words():
    f = open('data/sentiment_words.txt', 'w', encoding='utf-8')
    with open('data/sentiment_words.csv', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            line = line.strip().replace(' ', '').split(',')
            if line[1] == 'idiom':
                continue
            if line[6] == '1.0':
                f.write(line[0] + ',' + str(line[5]) + '\n')
            elif line[6] == '2.0':
                f.write(line[0] + ',' + str(-1 * float(line[5])) + '\n')
    f.close()


def normalize_sentiment_words():
    words = []
    weights = []
    with open('data/sentiment_words.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')
            words.append(line[0])
            weights.append(float(line[1]))
    weights = np.array(weights)
    mean = weights.mean()
    std = weights.std()
    weights = (weights - mean)/std
    with open('data/normal_sentiment_words.txt', 'w', encoding='utf-8') as fp:
        for i in range(len(words)):
            fp.write(words[i] + ',' + str(weights[i]) + '\n')


def process_words_list():
    stopwords = get_stopwords()
    sentiment_dict = get_sentiment_dict()
    sentences = []
    words_list = []
    with open('data/positive.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())
    with open('data/negative.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())
    jieba.load_userdict(sentiment_dict.keys())
    for sentence in sentences:
        sentence = list(jieba.cut(sentence))
        for word in sentence:
            if '\u4e00' <= word <= '\u9fff' and word not in stopwords:
                words_list.append(word)
    with open('data/word_list.txt', 'w', encoding='utf-8') as fp:
        for word in words_list:
            fp.write(word + '\n')


def get_words_list():
    words_list = []
    with open('data/word_list.txt', 'w', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            words_list.append(line.strip())
    return words_list


def get_stopwords():
    stopwords = []
    with open('data/stop.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def get_sentiment_dict():
    sentiment_dict = {}
    with open('data/normal_sentiment_words.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')
            sentiment_dict[line[0]] = line[1]
    return sentiment_dict


def process_word_vectors():
    from bert_serving.client import BertClient
    bc = BertClient()
    word_list = get_words_list()
    vecs = []
    for word in word_list:
        vec = bc.encode([word])
        vecs.append(vec[0])
    vecs = np.array(vecs)
    np.savetxt("data/vecs.txt", vecs)


def get_word_vectors():
    word_list = get_words_list()
    vecs = np.loadtxt('data/vecs.txt')
    word2vec = {}
    for i in range(len(word_list)):
        word2vec[word_list[i]] = vecs[i]
    return word2vec
