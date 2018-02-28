import torch
import torch.autograd as autograd
from torch.autograd.variable import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False, debug=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.debug = debug

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        print(f"generator.forward(inp, hidden={hidden}")
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        # out = F.log_softmax(out)
        out = F.log_softmax(out, dim=-1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        print(f"generator.sample(num_samples = {num_samples}, start_letter={start_letter}")
        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)   # shape = torch.Size([32, 20])
        h = self.init_hidden(num_samples)

        _bt = [start_letter] * num_samples
        _base = torch.cuda.LongTensor(_bt) if self.gpu else torch.LongTensor(_bt)
        inp = autograd.Variable(_base)

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            input = torch.exp(out)
            if self.debug: print(f"input.size = {input.size()}")
            out = torch.multinomial(input, 1)               # num_samples x 1 (sampling from each row)

            # TODO: Fix size mismatch here
            # out_np = out.data.cpu().numpy()
            samples[:, i] = out.data                    #out.data.shape = torch.Size([32, 1])

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        print(f"generator.batchNLLLoss(inp={inp}, target={target}")
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

