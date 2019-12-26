# import jieba
# from lxml import etree
# import numpy as np
from bert_serving.client import BertClient


# def ZscoreNormalization(x):
#     """Z-score normaliaztion"""
#     x = (x - np.mean(x)) / np.std(x)
#     return x
#
#
# max_features = 25000
# max_len = 300
# batch_size = 32


if __name__ == '__main__':
    bc = BertClient()  # ip address of the GPU machine
    print(bc.encode(['最']))
    # bc = BertClient()
    # stopwords = []
    # sentiments = []
    # # intensity = []
    # sentivalue = {}
    # with open('stopword.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         stopwords.append(line.strip())
    # with open('sentiments.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split(',')
    #         sentiments.append(line[0])
    #         # intensity.append(line[1])
    #         sentivalue[line[0]] = float(line[1])
    # jieba.load_userdict(sentiments)
    # with open('sample.positive.txt', 'r', encoding='utf-8') as f:
    #     content = f.read()
    #     content = etree.HTML(content)
    #     reviews = content.xpath('//review/text()')[-1:]
    #     data = []
    #     for review in reviews:
    #         data.append(review.strip().replace('\n', '').replace('\u3000', '').replace(' ', ''))
    #     sentences = []
    #     for dt in data:
    #         dt = list(jieba.cut(dt))
    #         # print(dt)
    #         sentence = []
    #         for word in dt:
    #             if word not in stopwords:
    #                 vec = bc.encode([word])
    #                 vec = vec[0]
    #                 if word in sentiments:
    #                     vec = sentivalue[word] * vec[0]
    #                 sentence.append(vec)
    #         # print(type(vec))
    #         # print(sentence)
    #         sentence = np.array(sentence)
    #         print(sentence)
    #         sentence = np.mat(sentence)
    #         print(sentence)
            # sentences.append(sentence)
            # sentences.append(dt)
        #     for word in dt:
        #         if word not in stopwords:
        #             print(word)
        #             sentence.append(word)
        #         sentences.append(sentence)
        # # print(sentences)
        # print(sentences)
            # print(dt)
    # file = open('sentiment.txt', 'w', encoding='utf-8')
    # with open('sentiment_words.csv', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split(', ')
    #         if line[1].strip() == 'adv' or line[1].strip() == 'adj' or line[1].strip() == 'noun' or \
    #                 line[1].strip() == 'verb' or line[1].strip() == 'prep' or line[1].strip() == 'nw':
    #             if line[6] == '1.0':
    #                 word = line[0].strip()
    #                 intensity = float(line[5])
    #                 file.write(word + ',' + str(intensity) + '\n')
    #             if line[6] == '2.0':
    #                 word = line[0].strip()
    #                 intensity = -float(line[5])
    #                 file.write(word + ',' + str(intensity) + '\n')
    # file.close()
    # intensity = []
    # words = []
    # with open('sentiment.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split(',')
    #         words.append(line[0])
    #         intensity.append(float(line[1]))
    # intensity = ZscoreNormalization(intensity)
    # news = []
    # for i in intensity:
    #     if i >= 0:
    #         i = i + 1
    #     else:
    #         i = i - 1
    #     news.append(i)
    # print(news)
    # file = open('sentiments.txt', 'w', encoding='utf-8')
    # for i in range(len(news)):
    #     file.write(words[i] + ',' + str(news[i]) + '\n')
    # file.close()

