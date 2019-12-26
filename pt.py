import torch as t
import torch.nn as nn
import numpy as np
import jieba
import re
import collections
from sklearn.model_selection import train_test_split
import torch.utils.data


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


class Net(nn.Module):
    def __init__(self, vocab_size, n_dim, sentence_length):
        super(Net, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, n_dim, padding_idx=0)
        pretrain_embeddings = np.loadtxt('vec_t.txt')
        self.word_embeddings.weight.data.copy_(t.from_numpy(pretrain_embeddings))
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(n_dim, 128, h),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(sentence_length - h + 1)
                           ) for h in [3, 4, 5]]
        )
        self.fc = nn.Linear(128 * 3, 2)

    def forward(self, x):
        embed_x = self.word_embeddings(x)
        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = t.cat(out, 1)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        return out


class MyData(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(new_sentences)

    def __getitem__(self, index):
        return label[index], np.array(new_sentences[index])


data_loader = torch.utils.data.DataLoader(MyData(), 32, True)


if __name__ == '__main__':
    net = Net(50000, 768, max_len)
    optimizer = t.optim.Adam(net.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for epoch in range(10):
        for i, (cls, sentences) in enumerate(data_loader):
            optimizer.zero_grad()
            sentences = sentences.type(t.LongTensor)
            cls = cls.type(t.LongTensor)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.numpy().tolist()
            pred = cls.numpy().tolist()
            # predict = predicted.numpy().tolist()
            # pred = cls.numpy().tolist()
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
            loss = criterion(out, cls)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)
