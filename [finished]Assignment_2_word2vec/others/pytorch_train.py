import random
import re

import torch
import torch.optim as optim
from tqdm import tqdm
from pytorch_word2vec_model import SkipGram

epochs = 50
negative_sampling = 4
window = 2
vocab_size = 1
embd_size = 300


def batch_data(x, batch_size=128):
    in_w = []
    out_w = []
    target = []
    for text in x:
        for i in range(window, len(text) - window):
            word_set = set()
            in_w.append(text[i])
            in_w.append(text[i])
            in_w.append(text[i])
            in_w.append(text[i])

            out_w.append(text[i - 2])
            out_w.append(text[i - 1])
            out_w.append(text[i + 1])
            out_w.append(text[i + 2])

            target.append(1)
            target.append(1)
            target.append(1)
            target.append(1)
            # negative sampling
            count = 0
            while count < negative_sampling:
                rand_id = random.randint(0, vocab_size-1)
                if not rand_id in word_set:
                    in_w.append(text[i])
                    out_w.append(rand_id)
                    target.append(0)
                    count += 1

            if len(out_w) >= batch_size:
                yield [in_w, out_w, target]
                in_w = []
                out_w = []
                target = []
    if out_w:
        yield [in_w, out_w, target]


def train(train_text_id, model,opt):
    model.train()  # 启用dropout和batch normalization
    ave_loss = 0
    pbar = tqdm()
    cnt=0
    for x_batch in batch_data(train_text_id):
        in_w, out_w, target = x_batch
        in_w_var = torch.tensor(in_w)
        out_w_var = torch.tensor(out_w)
        target_var = torch.tensor(target,dtype=torch.float)

        model.zero_grad()
        log_probs = model(in_w_var, out_w_var)
        loss = model.loss(log_probs, target_var)
        loss.backward()
        opt.step()
        ave_loss += loss.item()
        pbar.update(1)
        cnt += 1
        pbar.set_description('< loss: %.5f >' % (ave_loss / cnt))
    pbar.close()
text_id = []
vocab_dict = {}

with open(
        'D:\\project\\ml\\github\\cs224n-natural-language-processing-winter2019\\a1_intro_word_vectors\\a1\\corpus\\corpus.txt',
        encoding='utf-8') as fp:
    for line in fp:
        lines = re.sub("[^A-Za-z0-9']+", ' ', line).lower().split()
        line_id = []
        for s in lines:
            if not s:
                continue
            if s not in vocab_dict:
                vocab_dict[s] = len(vocab_dict)
            id = vocab_dict[s]
            line_id.append(id)
            if id==11500:
                print(id,s)
        text_id.append(line_id)
vocab_size = len(vocab_dict)
print('vocab_size', vocab_size)
model = SkipGram(vocab_size, embd_size)

for epoch in range(epochs):
    print('epoch', epoch)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, weight_decay=0)
    train(text_id, model,opt)

