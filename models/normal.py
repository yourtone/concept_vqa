import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import MFH, GatedTanh


class V2V(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(V2V, self).__init__()
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.wedp = nn.Dropout(0.5)
        self.gru = nn.GRU(input_size=emb_size,
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


class MultiAttModel(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(MultiAttModel, self).__init__()
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.wedp = nn.Dropout(0.5)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.5)
        self.que_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 2*512),
                nn.Tanh())

        self.img_same_net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv1d(2048, 512, kernel_size=1),
                nn.Tanh())
        self.que_same_net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.Tanh())

        self.att_w_fc = nn.Conv1d(512, 2, kernel_size=1)

        self.att_img1_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.Tanh())
        self.att_img2_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.Tanh())

        self.pred_net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2*512, num_ans))

    def forward(self, img, que):
        bs = img.size()[0]
        emb = self.wedp(self.we(que))
        _, hn = self.gru(emb)
        hn = hn.squeeze(dim=0)

        # final question feature
        que_fea = self.que_fc(hn)

        # final image feature
        img = img.transpose(1, 2)
        img_norm = F.normalize(img, p=2, dim=1)

        ## attention
        img_same = self.img_same_net(img_norm)  # b x 512 x 36
        que_same = self.que_same_net(hn).unsqueeze(dim=2).expand(bs, 512, 36)
        att_w1, att_w2 = self.att_w_fc(torch.mul(img_same, que_same)).split(1, dim=1)

        att_w1 = F.softmax(att_w1.squeeze(dim=1))
        att_w1_exp = att_w1.unsqueeze(dim=1).expand_as(img_norm)
        att_img1 = torch.mul(img_norm, att_w1_exp).sum(dim=2)
        att_img1 = self.att_img1_fc(att_img1)

        att_w2 = F.softmax(att_w2.squeeze(dim=1))
        att_w2_exp = att_w2.unsqueeze(dim=1).expand_as(img_norm)
        att_img2 = torch.mul(img_norm, att_w2_exp).sum(dim=2)
        att_img2 = self.att_img2_fc(att_img2)

        img_fea = torch.cat((att_img1, att_img2), dim=1)

        # predict answer
        mul_fea = torch.mul(que_fea, img_fea)
        score = self.pred_net(mul_fea)

        return score


class GatedMultiAttModel(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(GatedMultiAttModel, self).__init__()
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.wedp = nn.Dropout(0.5)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.5)
        self.que_fc = nn.Sequential(
                nn.Dropout(0.5),
                GatedTanh(512, 2*512))

        self.img_same_net = nn.Sequential(
                nn.Dropout(0.5),
                GatedTanh(2048, 512, use_conv=True))
        self.que_same_net = nn.Sequential(
                nn.Dropout(0.5),
                GatedTanh(512, 512))

        self.att_w_fc = nn.Conv1d(512, 2, kernel_size=1)

        self.att_img1_fc = nn.Sequential(
                nn.Dropout(0.5),
                GatedTanh(2048, 512))
        self.att_img2_fc = nn.Sequential(
                nn.Dropout(0.5),
                GatedTanh(2048, 512))

        self.pred_net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2*512, num_ans))

    def forward(self, img, que):
        bs = img.size()[0]
        emb = self.wedp(self.we(que))
        _, hn = self.gru(emb)
        hn = hn.squeeze(dim=0)

        # final question feature
        que_fea = self.que_fc(hn)

        # final image feature
        img = img.transpose(1, 2)
        img_norm = F.normalize(img, p=2, dim=1)

        ## attention
        img_same = self.img_same_net(img_norm)  # b x 512 x 36
        que_same = self.que_same_net(hn).unsqueeze(dim=2).expand(bs, 512, 36)
        att_w1, att_w2 = self.att_w_fc(torch.mul(img_same, que_same)).split(1, dim=1)

        att_w1 = F.softmax(att_w1.squeeze(dim=1))
        att_w1_exp = att_w1.unsqueeze(dim=1).expand_as(img_norm)
        att_img1 = torch.mul(img_norm, att_w1_exp).sum(dim=2)
        att_img1 = self.att_img1_fc(att_img1)

        att_w2 = F.softmax(att_w2.squeeze(dim=1))
        att_w2_exp = att_w2.unsqueeze(dim=1).expand_as(img_norm)
        att_img2 = torch.mul(img_norm, att_w2_exp).sum(dim=2)
        att_img2 = self.att_img2_fc(att_img2)

        img_fea = torch.cat((att_img1, att_img2), dim=1)

        # predict answer
        mul_fea = torch.mul(que_fea, img_fea)
        score = self.pred_net(mul_fea)

        return score


class MFHModel(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(MFHModel, self).__init__()
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True)
        self.grudp = nn.Dropout(0.3)
        self.att_mfh = MFH(2048, 512, latent_dim=4,
                           output_size=1024, block_count=2)
        self.att_net = nn.Sequential(
                nn.Linear(1024*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh = MFH(2048, 512, latent_dim=4,
                            output_size=1024, block_count=2)
        self.pred_net = nn.Linear(1024*2, num_ans)

    def forward(self, img, que):
        emb = F.tanh(self.we(que))
        _, hn = self.gru(emb)
        hn = self.grudp(hn).squeeze(dim=0)
        img_norm = F.normalize(img, p=2, dim=2)

        att_w = self.att_net(self.att_mfh(img_norm, hn))
        att_w_exp = F.softmax(att_w, dim=1).permute(0, 2, 1)
        att_img = torch.bmm(att_w_exp, img_norm)
        att_img = att_img.view(att_img.size(0), -1)

        score = self.pred_net(self.pred_mfh(att_img, hn))

        return score

