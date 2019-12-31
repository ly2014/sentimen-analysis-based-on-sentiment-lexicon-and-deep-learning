import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jieba
import re
import collections
from sklearn.model_selection import train_test_split
import torch.utils.data
from data_loader import MyData
from model import SLCABG
import data_util


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


def load_data(filename):
    rs_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('[^\u4e00-\u9fa5]', '', line)
            rs_data.append(line)
    return rs_data


max_len = 648
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

vecs = np.loadtxt('vec_t.txt')
words = []
with open('word_t.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        words.append(line.strip())
new_vecs = []
for i in range(len(words)):
    if words[i] in sentivalue.keys():
        new_vecs.append(vecs[i] * sentivalue[words[i]])
    else:
        new_vecs.append(vecs[i])
new_vecs = np.array(new_vecs)

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
for sentence in sentences:
    for word in sentence:
        frequency[word] += 1
word2index = dict()
for i, x in enumerate(frequency.most_common(50000)):
    word2index[x[0]] = i + 1
new_sentences = []
for sentence in sentences:
    sen = []
    for word in sentence:
        if word in word2index.keys():
            sen.append(word2index[word])
        else:
            sen.append(0)
    if len(sen) < max_len:
        sen.extend([0 for i in range(max_len - len(sen))])
    else:
        sen = sen[:max_len]
    new_sentences.append(sen)
x_train, x_test, y_train, y_test = train_test_split(new_sentences, label, test_size=0.2)

train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)


if __name__ == '__main__':
    net = SLCABG(50000, 768, max_len).to(device)
    optimizer = t.optim.Adam(net.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    tp = 1
    tn = 1
    fp = 1
    fn = 1
    for epoch in range(15):
        for i, (cls, sentences) in enumerate(train_data_loader):
            optimizer.zero_grad()
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                # print(f, n)
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            loss = criterion(out, cls).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)

    # Test the model
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    print('==========================================================================================')
    with torch.no_grad():
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for cls, sentences in test_data_loader:
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
        print('acc', acc, 'p', p, 'r', r, 'f1', f1)
