import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super(GraphConv, self).__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, mask):
        super(GCN, self).__init__()

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        #self.adj = adj.cuda()
        self.adj = adj
        self.mask = mask
        self.MLP = nn.Linear(out_channels+out_channels, 1)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels  # dim.input
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def add_atten_dot(self, inputs):
        # dot
        # output = inputs * self.mask
        output = inputs
        print output
        output_T = output.t()
        logits = torch.mm(output, output_T)
        coefs = F.softmax(logits, dim=1)
        # print coefs
        output_atten = torch.mm(coefs, inputs)
        return output_atten

    def add_atten_cos1(self, inputs):
        output = inputs
        # print output

        # consin distance
        output = F.normalize(output)
        output_T = output.t()
        logits = torch.mm(output, output_T)

        logits = logits * self.mask
        logits = logits * self.mask.t()
        coefs = F.normalize(logits, dim=1)
        # coefs = F.softmax(logits, dim=1)
        # coefs = logits
        # print coefs

        output_atten = torch.matmul(coefs, inputs)
        return output_atten

    def add_atten_cos_awa(self, inputs):
        output = inputs
        # print output

        # consin distance
        output = F.normalize(output)
        output_T = output.t()
        logits = torch.mm(output, output_T)

        # mask = self.mask
        # mask /= torch.mean(mask)
        # print self.mask
        logits = logits * self.mask

        # logits = logits * self.mask.t()
        coefs = F.softmax(logits, dim=1)
        # coefs = F.softmax(logits, dim=1)
        # coefs = logits
        # print coefs
        coefs = torch.where(coefs < 1e-3, torch.full_like(coefs, 0), coefs)
        # print coefs
        output_atten = torch.mm(coefs, inputs)
        # print output_atten
        return output_atten

    def add_atten_cos(self, inputs):
        output = inputs
        # print output

        # consin distance
        output = F.normalize(output)
        output_T = output.t()
        logits = torch.mm(output, output_T)

        # mask = self.mask
        # mask /= torch.mean(mask)
        # print self.mask
        logits = logits * self.mask

        # logits = logits * self.mask.t()
        coefs = F.softmax(logits, dim=1)
        # coefs = F.softmax(logits, dim=1)
        # coefs = logits
        # print coefs
        coefs = torch.where(coefs < 1e-3, torch.full_like(coefs, 0), coefs)
        # print coefs
        output_atten = torch.mm(coefs, inputs)
        # print output_atten
        return output_atten, coefs

    def add_atten_MLP(self, inputs):

        feats = inputs


        weights = list()
        for i in range(len(feats)):
            feat_i = feats[i]
            feat_i = feat_i.expand(len(feats), feat_i.shape[0])
            x = torch.cat((feat_i, feats), 1)

            weight_i = self.MLP(x)
            # norm_weisghts = F.softmax(torch.cat(weights, -1), dim=-1)
            weight_i = torch.squeeze(weight_i.t())
            print i, ' weight ', weight_i.shape
            weights.append(weight_i)

        logits = torch.stack(weights, dim=0)
        coefs = logits * self.mask
        print coefs

        output_atten = torch.mm(coefs, inputs)
        return output_atten

    def forward(self, x):

        for conv in self.layers:
            x = conv(x, self.adj)

        x = F.normalize(x)

        x_atten, coefs = self.add_atten_cos(x)
        return x_atten, coefs
        # return x

