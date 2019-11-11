import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus)
        embed_ctx = self.embeddings(context)
        # score = torch.mm(embed_focus, torch.t(embed_ctx))
        score = torch.mul(embed_focus, embed_ctx).sum(dim=1)
        log_probs = score #F.logsigmoid(score)

        return log_probs

    def loss(self, log_probs, target):
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = nn.NLLLoss()
        loss = loss_fn(log_probs, target)
        return loss


class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2 * context_size * embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out)
        return log_probs
