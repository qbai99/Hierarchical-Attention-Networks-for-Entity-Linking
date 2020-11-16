"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class MyWordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size):
        super(MyWordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep="\t", quoting=csv.QUOTE_NONE, usecols=[2]).values

        dict_len, embed_size = dict.shape
        dict_len += 1
        embed_size = 300
        vec = np.zeros((dict_len, embed_size))
        test = open(word2vec_path, encoding='utf8').readlines()
        for i in range(len(test)):
            first = test[i]
            tmp = first.split()[2].split(',')
            j = 0
            for vec_str in tmp:
                vec[i, j] = float(vec_str)
                j = j + 1
        dict=vec
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
        self.dict = dict
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        # 从数据字典中获取词向量
        output = self.lookup(input)
        # Word Encoder 放入gru feature output and hidden state output，
        f_output, h_output = self.gru(output.float(), hidden_state)
        # 增加tone-layer MLP tanh()  get hidden representation
        output = torch.tanh(matrix_mul(f_output, self.word_weight, self.word_bias))
        # a word level context vector
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        # 归一化得到权重矩阵 importance weight ait through a softmax function
        output = F.softmax(output, dim=1)
        # 通过attention权重矩阵，将句子向量看作组成这些句子的词向量的加权求和。
        output = element_wise_mul(f_output,output.permute(1, 0))
        return output, h_output


if __name__ == "__main__":
    abc = MyWordAttNet("../data/word_info.txt", hidden_size=50)



