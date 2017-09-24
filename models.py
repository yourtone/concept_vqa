import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, num_words, num_ans):
        super(Baseline, self).__init__()
        self.we = nn.Embedding(num_words, 300, padding_idx=0)
        self.wedp = nn.Dropout(0.5)
        self.gru = nn.GRU(input_size=300,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.5)
        self.att_net = nn.Sequential(
            nn.Conv1d(2048+512, 512, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 1, kernel_size=1))

        self.que_fc1 = nn.Linear(512, 512)
        self.img_fc1 = nn.Linear(2048, 512)
        self.pred_net = nn.Sequential(
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(512, num_ans))

    def forward(self, img, que):
        bs = img.size()[0]
        emb = self.wedp(self.we(que))
        _, hn = self.gru(emb)
        hn = hn.squeeze(dim=0)

        img = img.transpose(1, 2)
        img_norm = F.normalize(img, p=2, dim=1)

        merge_fea = torch.cat((hn.unsqueeze(dim=2).expand(bs, 512, 36),
                               img_norm),
                              dim=1)
        att_w = F.softmax(self.att_net(merge_fea).squeeze(dim=1))
        att_w_exp = att_w.unsqueeze(dim=1).expand_as(img_norm)
        att_img = torch.mul(img_norm, att_w_exp).sum(dim=2)

        que_fea1 = F.tanh(self.que_fc1(hn))
        img_fea1 = F.tanh(self.img_fc1(att_img))
        mul_fea = torch.mul(que_fea1, img_fea1)
        score = self.pred_net(mul_fea)

        return score

