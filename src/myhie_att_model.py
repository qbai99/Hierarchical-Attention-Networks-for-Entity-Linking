"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
import csv
import pandas as pd
from src.mysen_att import MySentAttNet
from src.myword_att import MyWordAttNet
import numpy as np


def sim(output, can_ent):
    num_output = output.shape
    res = torch.rand((num_output))
    for iter, c in enumerate(can_ent):
        ent_emebed=output[iter].unsqueeze(0)
        sim_res_list = []
        for c0 in c[0]:
            sim_res = torch.cosine_similarity(ent_emebed, c0.unsqueeze(0))
            sim_res_list.append(sim_res)
        res[iter] = torch.tensor(sim_res_list)
    return res


class MyHierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size,  pretrained_word2vec_path, ent_path,
                 max_sent_length, max_word_length):
        super(MyHierAttNet, self).__init__()

        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = MyWordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = MySentAttNet(sent_hidden_size, word_hidden_size)

        self.lin = torch.nn.Linear(600, 300)
        self._init_hidden_state()
        env = pd.read_csv(filepath_or_buffer=ent_path, header=None, sep="\t", quoting=csv.QUOTE_NONE,
                           usecols=[1]).values
        # dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
        #                    usecols=[1]).values
        env_len, embed_size = env.shape
        env_len += 1
        embed_size = 300
        vec = np.zeros((env_len, embed_size))
        test = open(ent_path, encoding='utf8').readlines()
        for i in range(len(test)):
            first = test[i]
            tmp = first.split()[1].split(',')
            j = 0
            for vec_str in tmp:
                vec[i, j] = float(vec_str)
                j = j + 1
        env = vec
        unknown_word = np.zeros((1, embed_size))
        env = torch.from_numpy(np.concatenate([unknown_word, env], axis=0).astype(np.float))
        self.env = env
        self.lookup = nn.Embedding(num_embeddings=env_len, embedding_dim=embed_size).from_pretrained(env)
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):
        # 读入数据
        docs = input[0]
        sens = input[1]
        candidates = input[2]
        idx = input[3]-1
        docs = docs.permute(1, 0, 2)
        # 候选实体表征向量查询
        outputdoc_list = []
        outputsen_list = []
        can_ent = self.lookup(candidates)
        # 获取文章表征
        for d in docs:
            output, self.word_hidden_state = self.word_att_net(d.permute(1, 0), self.word_hidden_state)
            outputdoc_list.append(output)
        output_per_sen = torch.cat(outputdoc_list, 0)#每句话的输出
        # 文章表征output_doc
        output_doc, self.sent_hidden_state = self.sent_att_net(output_per_sen, self.sent_hidden_state)
        # 获取句子特征向量
        output_tmp = output_per_sen.permute(1, 0, 2)
        for iter, sen in enumerate(output_tmp):
            outputsen_list.append(sen[sens[iter]].unsqueeze(0))
        # 句子表征output_doc
        output_sen = torch.cat(outputsen_list, 0)
        # 获取句子，文章层面与候选实体的相似度
        sim_sen = sim(output_sen, can_ent)
        sim_doc = sim(output_doc, can_ent)
        sim_total = torch.cat((sim_sen, sim_doc), 1)
        # 使用MLP 获取所有候选实体得分
        score = self.lin(sim_total)
        # 找到正确链接的得分
        sc_list = []
        for iter, s in enumerate(score):
            sc_list.append(s[idx[iter]].unsqueeze(0))
        score_gold = torch.cat(sc_list, 0)
        # 找到得分最高的候选实体的索引
        max_index = torch.argmax(score, dim=1)
        return score_gold, score, max_index

if __name__ == "__main__":
    abc = MyHierAttNet(word_hidden_size=50,sent_hidden_size=50,batch_size=128,max_sent_length=10,max_word_length=150,pretrained_word2vec_path="../data/word_info.txt",ent_path="../data/ent_vec.txt")

