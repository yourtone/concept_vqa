import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MFH


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

        self.att_mfh2 = MFH(300, 512, latent_dim=4,
                           output_size=1024, block_count=2)
        self.att_net2 = nn.Sequential(
                nn.Linear(1024*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh2 = MFH(300, 512, latent_dim=4,
                            output_size=1024, block_count=2)
        self.pred_net2 = nn.Linear(1024*2, num_ans)

    def forward(self, img, que, ocr):
        emb = F.tanh(self.we(que))
        _, hn = self.gru(emb)
        hn = self.grudp(hn).squeeze(dim=0)
        img_norm = F.normalize(img, p=2, dim=2)
        ocr_norm = F.normalize(ocr, p=2, dim=2)

        att_w = self.att_net(self.att_mfh(img_norm, hn))
        att_w_exp = F.softmax(att_w, dim=1).permute(0, 2, 1)
        att_img = torch.bmm(att_w_exp, img_norm)
        att_img = att_img.view(att_img.size(0), -1)

        att_w2 = self.att_net2(self.att_mfh2(ocr_norm, hn))
        att_w_exp2 = F.softmax(att_w2, dim=1).permute(0, 2, 1)
        att_ocr = torch.bmm(att_w_exp2, ocr_norm)
        att_ocr = att_ocr.view(att_ocr.size(0), -1)

        score = self.pred_net(self.pred_mfh(att_img, hn))

        score2 = self.pred_net2(self.pred_mfh2(att_ocr, hn))

        return score+score2


class MFHModela(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(MFHModela, self).__init__()
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
        self.pred_net = nn.Sequential(
                nn.Linear(1024*2, emb_size),
                nn.Tanh())
        self.ans_emb_net = nn.Linear(emb_size, num_ans, bias=False)

        self.att_mfh2 = MFH(300, 512, latent_dim=4,
                           output_size=1024, block_count=2)
        self.att_net2 = nn.Sequential(
                nn.Linear(1024*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh2 = MFH(300, 512, latent_dim=4,
                            output_size=1024, block_count=2)
        self.pred_net2 = nn.Sequential(
                nn.Linear(1024*2, emb_size),
                nn.Tanh())
        self.ans_emb_net2 = nn.Linear(emb_size, num_ans, bias=False)

    def forward(self, img, que, ocr):
        emb = F.tanh(self.we(que))
        _, hn = self.gru(emb)
        hn = self.grudp(hn).squeeze(dim=0)
        img_norm = F.normalize(img, p=2, dim=2)
        ocr_norm = F.normalize(ocr, p=2, dim=2)

        att_w = self.att_net(self.att_mfh(img_norm, hn))
        att_w_exp = F.softmax(att_w, dim=1).permute(0, 2, 1)
        att_img = torch.bmm(att_w_exp, img_norm)
        att_img = att_img.view(att_img.size(0), -1)

        att_w2 = self.att_net2(self.att_mfh2(ocr_norm, hn))
        att_w_exp2 = F.softmax(att_w2, dim=1).permute(0, 2, 1)
        att_ocr = torch.bmm(att_w_exp2, ocr_norm)
        att_ocr = att_ocr.view(att_ocr.size(0), -1)

        fuse_emb = self.pred_net(self.pred_mfh(att_img, hn))
        #fuse_emb = F.normalize(fuse_emb, p=2, dim=1)
        score = self.ans_emb_net(fuse_emb)

        fuse_emb2 = self.pred_net2(self.pred_mfh2(att_ocr, hn))
        #fuse_emb2 = F.normalize(fuse_emb2, p=2, dim=1)
        score2 = self.ans_emb_net2(fuse_emb2)

        return score+score2


class MFHModelc(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(MFHModelc, self).__init__()
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
        self.pred_net = nn.Sequential(
                nn.Linear(1024*2, emb_size),
                nn.Tanh())

        self.att_mfh2 = MFH(300, 512, latent_dim=4,
                           output_size=1024, block_count=2)
        self.att_net2 = nn.Sequential(
                nn.Linear(1024*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh2 = MFH(300, 512, latent_dim=4,
                            output_size=1024, block_count=2)
        self.pred_net2 = nn.Sequential(
                nn.Linear(1024*2, emb_size),
                nn.Tanh())

    def forward(self, img, que, ocr):
        emb = F.tanh(self.we(que))
        _, hn = self.gru(emb)
        hn = self.grudp(hn).squeeze(dim=0)
        img_norm = F.normalize(img, p=2, dim=2)
        ocr_norm = F.normalize(ocr, p=2, dim=2)

        att_w = self.att_net(self.att_mfh(img_norm, hn))
        att_w_exp = F.softmax(att_w, dim=1).permute(0, 2, 1)
        att_img = torch.bmm(att_w_exp, img_norm)
        att_img = att_img.view(att_img.size(0), -1)

        att_w2 = self.att_net2(self.att_mfh2(ocr_norm, hn))
        att_w_exp2 = F.softmax(att_w2, dim=1).permute(0, 2, 1)
        att_ocr = torch.bmm(att_w_exp2, ocr_norm)
        att_ocr = att_ocr.view(att_ocr.size(0), -1)

        fuse_emb = self.pred_net(self.pred_mfh(att_img, hn))
        # score = torch.bmm(ocr, fuse_emb.unsqueeze(2)).squeeze()

        fuse_emb2 = self.pred_net2(self.pred_mfh2(att_ocr, hn))
        # score2 = torch.bmm(ocr, fuse_emb2.unsqueeze(2)).squeeze()

        return fuse_emb+fuse_emb2


class DiffAttModelc(nn.Module):
    def __init__(self, num_words, num_ans, emb_size):
        super(DiffAttModelc, self).__init__()
        self.we = nn.Embedding(num_words, emb_size, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=512,
                          num_layers=1,
                          batch_first=True)
        self.grudp = nn.Dropout(0.3)
        self.att_mfh1 = MFH(2048, 512, latent_dim=4,
                           output_size=512, block_count=2)
        self.att_net1_1 = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))
        self.att_net1_2 = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh1_1 = MFH(2048, 512, latent_dim=4,
                            output_size=512, block_count=2)
        self.pred_mfh1_2 = MFH(2048, 512, latent_dim=4,
                            output_size=512, block_count=2)
        self.pred_net1_1 = nn.Sequential(
                nn.Linear(512*2, emb_size),
                nn.Tanh())
        self.pred_net1_2 = nn.Sequential(
                nn.Linear(512*2, emb_size),
                nn.Tanh())

        self.att_mfh2 = MFH(300, 512, latent_dim=4,
                           output_size=512, block_count=2)
        self.att_net2_1 = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))
        self.att_net2_2 = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1))

        self.pred_mfh2_1 = MFH(300, 512, latent_dim=4,
                            output_size=512, block_count=2)
        self.pred_mfh2_2 = MFH(300, 512, latent_dim=4,
                            output_size=512, block_count=2)
        self.pred_net2_1 = nn.Sequential(
                nn.Linear(512*2, emb_size),
                nn.Tanh())
        self.pred_net2_2 = nn.Sequential(
                nn.Linear(512*2, emb_size),
                nn.Tanh())

    def forward(self, img, que, ocr):
        emb = F.tanh(self.we(que))
        _, hn = self.gru(emb)
        hn = self.grudp(hn).squeeze(dim=0)
        img_norm = F.normalize(img, p=2, dim=2)
        ocr_norm = F.normalize(ocr, p=2, dim=2)

        att_fuse_fea1 = self.att_mfh1(img_norm, hn)
        att_w1_1 = self.att_net1_1(att_fuse_fea1)
        att_w_exp1_1 = F.softmax(att_w1_1, dim=1).permute(0, 2, 1)
        att_img_1 = torch.bmm(att_w_exp1_1, img_norm)
        att_img_1 = att_img_1.view(att_img_1.size(0), -1)
        att_w1_2 = self.att_net1_2(att_fuse_fea1)
        att_w_exp1_2 = F.softmax(att_w1_2, dim=1).permute(0, 2, 1)
        att_img_2 = torch.bmm(att_w_exp1_2, img_norm)
        att_img_2 = att_img_2.view(att_img_2.size(0), -1)

        fuse_emb1 = self.pred_net1_1(self.pred_mfh1_1(att_img_1, hn)) + self.pred_net1_2(self.pred_mfh1_2(att_img_2, hn))

        att_fuse_fea2 = self.att_mfh2(ocr_norm, hn)
        att_w2_1 = self.att_net2_1(att_fuse_fea2)
        att_w_exp2_1 = F.softmax(att_w2_1, dim=1).permute(0, 2, 1)
        att_ocr_1 = torch.bmm(att_w_exp2_1, ocr_norm)
        att_ocr_1 = att_ocr_1.view(att_ocr_1.size(0), -1)
        att_w2_2 = self.att_net2_2(att_fuse_fea2)
        att_w_exp2_2 = F.softmax(att_w2_2, dim=1).permute(0, 2, 1)
        att_ocr_2 = torch.bmm(att_w_exp2_2, ocr_norm)
        att_ocr_2 = att_ocr_2.view(att_ocr_2.size(0), -1)

        fuse_emb2 = self.pred_net2_1(self.pred_mfh2_1(att_ocr_1, hn)) + self.pred_net2_2(self.pred_mfh2_2(att_ocr_2, hn))

        return fuse_emb1 + fuse_emb2

