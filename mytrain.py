"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.mydataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from src.myhie_att_model import MyHierAttNet


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epoches", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--word_hidden_size", type=int, default=150)
    parser.add_argument("--sent_hidden_size", type=int, default=150)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/documents_train.json")
    parser.add_argument("--test_set", type=str, default="data/documents_test.json")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/word_info.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--ent_path", type=str, default="data/ent_vec.txt")
    parser.add_argument("--stopwords_path", type=str, default="data/stopword.txt")
    args = parser.parse_args()
    return args


def train(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    print('logs opened')
    print("Model's parameters: {}".format(vars(opt)))
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    max_sent_length = 10
    max_word_length = 130
    training_set = MyDataset(opt.train_set, opt.word2vec_path, opt.ent_path, opt.stopwords_path, max_sent_length,
                             max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.word2vec_path, opt.ent_path, opt.stopwords_path, max_sent_length,
                         max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    model = MyHierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size,
                         opt.word2vec_path, opt.ent_path, max_sent_length, max_word_length)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()
    #修改loss函数
    criterion = nn.MarginRankingLoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    #修改训练部分代码
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        score_train = 0
        for iter, (doc, sen, candidates, gold_index) in enumerate(training_generator):  # mydataset 的返回值
                if torch.cuda.is_available():
                    doc = doc.cuda()
                    sen = sen.cuda()
                    candidates = candidates.cuda()
                    gold_index = gold_index.cunda()
                optimizer.zero_grad()
                model._init_hidden_state()
                batch_traindata = (doc, sen, candidates, gold_index)
                #修改模型输出
                score_gold, score, predictions = model(batch_traindata)
                #修改loss的计算方式
                tr_loss_ls = []
                for iter_, s in enumerate(score):
                    score_this_gold = torch.ones(s.shape)
                    score_this_gold[:] = score_gold[iter_]
                    ylabel = torch.ones(score_this_gold.shape)
                    loss_1 = criterion(s, score_this_gold, ylabel)
                    tr_loss_ls.append(loss_1)
                loss = sum(tr_loss_ls)/len(gold_index)
                loss.backward()
                optimizer.step()
                score = int(sum(gold_index == predictions))

                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Pre:{} Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, predictions, score / opt.batch_size))
                # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                # writer.add_scalar('Train/Accuracy', score//opt.batch_size, epoch * num_iter_per_epoch + iter)
                score_train += score
        print("Epoch: {}/{}======================> Epoch:Train_Accuracy: {}".format(
            epoch + 1,
            opt.num_epoches,
            score_train / (opt.batch_size * num_iter_per_epoch)))
        #修改测试部分代码
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            k = 0
            for te_doc, te_sen, te_candidates, te_gold_index in test_generator:
                num_sample = len(te_gold_index)
                if torch.cuda.is_available():
                    te_doc = te_doc.cuda()
                    te_sen = te_sen.cuda()
                    te_candidates = te_candidates.cuda()
                    te_gold_index = te_gold_index.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    #修改模型输出
                    te_score_gold, te_score, te_predictions = model((te_doc, te_sen, te_candidates, te_gold_index))
                te_loss_ls = []
                #修改loss的计算方式
                for iter__, te_s in enumerate(te_score):
                    te_score_this_gold = torch.ones(s.shape)
                    te_score_this_gold[:] = te_score_gold[iter__]
                    te_ylabel = torch.ones(te_score_this_gold.shape)
                    te_loss_1 = criterion(te_score_this_gold, te_s, te_ylabel)
                    te_loss_ls.append(te_loss_1)
                te_loss_tmp = sum(tr_loss_ls) / len(gold_index)
                loss_ls.append(te_loss_tmp * num_sample)
                te_label_ls.append(te_gold_index.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
                k=k+1
                if k == 57:
                    break
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = torch.cat(te_label_ls, 0)
            te_score = int(sum(te_pred == te_label))
            len_test = test_set.__len__()-1

            print("Epoch: {}/{}======================> Epoch:Test_Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                te_score / len_test))

            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = get_args()
    train(opt)
