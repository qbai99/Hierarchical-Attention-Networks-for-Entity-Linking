"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import json

import pandas as pd
from torch.utils.data.dataset import Dataset
from src.utils import get_max_lengths, get_evaluation
import csv
from nltk.tokenize import word_tokenize
import numpy as np
import re


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, ent_path, stopwords_path, max_length_sentences, max_length_word):
        super(MyDataset, self).__init__()
        sentences = []
        ids = []
        texts, labels = [], []
        mentions = []
        mention_sens, candidates, gold_indexes = [], [], []
        with open(data_path, 'r', encoding='utf8') as fp:
            train_data = json.load(fp)
        for document in train_data['documents']:
            for mention in document['mentions']:
                text = document['document']
                id = document['id']
                candidates_ = []
                mention_sen = int(mention['sent_index'])
                candidate = mention['candidates'].split('\t')
                for i in range(len(candidate)):
                    if i % 2 != 0:
                        candidates_.append(int(candidate[i]))
                sens = sent_tokenize(text)
                gold_index = int(mention['gold_index'])
                candidates.append(candidates_)
                gold_indexes.append(gold_index)
                ids.append(id)
                labels.append(int(candidates_[gold_index - 1]))
                mention_sens.append(mention_sen)
                texts.append(text)

        # train_data_=id+mention_sens+candidates+gold_indexes
        # self.train_data_=train_data_
        self.ids = ids
        self.sens = sens
        self.mention_sens = mention_sens
        self.candidates = candidates
        self.gold_indexes = gold_indexes
        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep="\t", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]

        self.env = pd.read_csv(filepath_or_buffer=ent_path, header=None, sep="\t", quoting=csv.QUOTE_NONE,
                               usecols=[0]).values
        self.env = [env[0] for env in self.env]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(self.labels)
        self.stopword = pd.read_csv(filepath_or_buffer=stopwords_path, header=None, sep="\t", quoting=csv.QUOTE_NONE,
                                    usecols=[0]).values
        self.stopword = [stopword[0] for stopword in self.stopword]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        max_candidates = 300
        stopword = self.stopword
        text = self.texts[index]
        candidates = self.candidates[index]
        gold_index = self.gold_indexes[index]
        mentionsen = self.mention_sens[index]
        # 对文本进行编码，过滤
        document_encode = [
            [self.dict.index(word) if word in self.dict and word not in stopword else -1 for word in
             word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text)]
        # 对实体进行编码，过滤
        candidates_encode = [self.env.index(i) if i in self.env else -1 for i in candidates]
        # 对文本长度进行补足，使其shape统一
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)
        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)
        # 对实体长度进行补足，使其shape统一
        if len(candidates_encode) < max_candidates:
            extended_candidate = [-1 for _ in range(max_candidates - len(candidates))]
            candidates_encode.extend(extended_candidate)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        candidates_encode = candidates_encode[:max_candidates]
        # 处理文本编码
        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        # 处理实体编码
        candidates_encode = np.array([candidates_encode])
        candidates_encode += 1
        # 返回文本编码， 句子编号， 实体编码， 正确实体编号
        return document_encode.astype(np.int64), mentionsen, candidates_encode.astype(np.int64), gold_index


def sent_tokenize(x):
    sents_temp = re.split('(：|:|,|，|。|！|\!|\.|？|\?)', x)
    sents = []
    for i in range(len(sents_temp) // 2):
        sent = sents_temp[2 * i] + sents_temp[2 * i + 1]
        sents.append(sent)
    return sents


if __name__ == '__main__':
    test = MyDataset(data_path="../data/documents_train.json", dict_path="../data/word_info.txt",
                     ent_path='../data/ent_vec.txt', stopwords_path='../data/stopword.txt', max_length_sentences = 10, max_length_word = 130)

    print (test.__getitem__(index=1)[0].shape)
