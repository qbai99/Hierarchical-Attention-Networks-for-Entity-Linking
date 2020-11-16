"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul

class MySentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        super(MySentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        # self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        # Sen Encoder 放入gru feature output and hidden state output，
        f_output, h_output = self.gru(input, hidden_state)
        # 增加tone-layer MLP tanh()  get hidden representation
        output = torch.tanh(matrix_mul(f_output, self.sent_weight, self.sent_bias))
        # a word sen context vector
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        # 归一化得到权重矩阵 importance weight ait through a softmax function
        output = F.softmax(output, dim=1)
        # 通过attention权重矩阵，将文本向量看作组成这些句子向量的加权求和。
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        # output = self.fc(output) #删除fc针对文本分类输出的处理

        return output, h_output


if __name__ == "__main__":
    abc = MySentAttNet()
