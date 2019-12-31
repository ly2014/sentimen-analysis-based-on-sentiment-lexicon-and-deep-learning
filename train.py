import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data
from data_loader import MyData
from model import SLCABG
import data_util


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
SENTENCE_LENGTH = 12
WORD_SIZE = 35000
EMBED_SIZE = 768


if __name__ == '__main__':
    sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)

    train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
    test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)

    net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH, word_vectors).to(device)
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

    net.eval()
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
