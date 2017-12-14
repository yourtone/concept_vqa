import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MFH(nn.Module):
    def __init__(self, x_size, y_size, latent_dim,
                 output_size, block_count, dropout=0.1):
        super(MFH, self).__init__()
        hidden_size = latent_dim * output_size
        self.x2hs = nn.ModuleList([
            nn.Linear(x_size, hidden_size) for i in range(block_count)])
        self.y2hs = nn.ModuleList([
            nn.Linear(y_size, hidden_size) for i in range(block_count)])
        self.dps = nn.ModuleList([
            nn.Dropout(dropout) for i in range(block_count)])

        self.latent_dim = latent_dim
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.block_count = block_count


    @staticmethod
    def align_dim(x, y):
        max_dims = x.size()
        if x.dim() > y.dim():
            diff_dim = [1,] * (x.dim() - y.dim())
            y_size = list(y.size())
            new_y_size = y_size[:1] + diff_dim + y_size[1:]
            y = y.view(*new_y_size)
            max_dims = x.size()
        elif x.dim() < y.dim():
            diff_dim = [1,] * (y.dim() - x.dim())
            x_size = x.size()
            new_x_size = x_size[:1] + diff_dim + x_size[1:]
            x = x.view(*new_x_size)
            max_dims = y.size()
        return x, y, list(max_dims)


    def forward(self, x, y):
        x, y, max_dims = self.align_dim(x, y)
        last_exp = Variable(torch.ones(self.hidden_size).type_as(x.data))
        exp_size = max_dims[:-1] + [self.output_size, self.latent_dim]
        results = []
        for i in range(self.block_count):
            xh = self.x2hs[i](x)
            yh = self.y2hs[i](y)
            last_exp = last_exp * self.dps[i](xh * yh)

            z_sum = last_exp.view(exp_size).sum(dim=-1)
            z_sqrt = z_sum.sign() * (z_sum.abs() + 1e-7).sqrt()
            z_norm = F.normalize(z_sqrt, p=2, dim=-1)

            results.append(z_norm)

        return torch.cat(results, dim=-1)


class GatedTanh(nn.Module):
    def __init__(self, in_size, out_size, bias=True, use_conv=False):
        super(GatedTanh, self).__init__()
        if use_conv:
            self.fc = nn.Conv1d(in_size, out_size, kernel_size=1, bias=bias)
            self.gate_fc = nn.Conv1d(in_size, out_size, kernel_size=1, bias=bias)
        else:
            self.fc = nn.Linear(in_size, out_size, bias=bias)
            self.gate_fc = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, input):
        return F.tanh(self.fc(input)) * F.sigmoid(self.gate_fc(input))

