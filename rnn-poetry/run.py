import re
import tqdm
import torch
import collections
import pickle
from torch import nn

from model import RNNModel,embed_size,hidden_dims,batch_size

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


epochs=100
lr=0.001
def get_data():
    special_character_removal = re.compile(r'[^\w。， ]', re.IGNORECASE)
    # 诗集
    poetrys = []
    peotry_path='data/poetry.txt'
    with open(peotry_path,'r',encoding='utf-8') as f:
        for content in f:
            content=content.strip()
            content = '[' + content + ']'
            poetrys.append(content)

    # poetrys = sorted(poetrys, key=lambda line: len(line))
    # 统计每个字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    # 取前多少个常用字
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID
    word2ix = dict(zip(words, range(len(words))))
    ix2word = {v: k for k, v in word2ix.items()}
    data = [[word2ix[c] for c in poetry ] for poetry in poetrys]
    # data=numpy.array(data)
    return data,word2ix,ix2word

def test(model):
    start_idx=[word2ix['[']]
    end_word=''
    lens=0
    hidden = None
    ret=''
    while end_word!=']' and len(ret)<100:
        data_ = torch.tensor([start_idx],device=device).long()
        # print("data size",data_.size())
        output, hidden = model(data_, hidden)
        # print("output size", output.size())
        ouput_idx=output.view(-1).argmax().cpu()
        # print('ouput_idx',ouput_idx)
        # print('ouput_idx', ouput_idx.item())
        ouput_idx=ouput_idx.item()
        start_idx=[ouput_idx]
        end_word=ix2word[ouput_idx]
        ret+=end_word
    return ret




def train():

    # 模型定义
    model = RNNModel(len(word2ix), embed_size, hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in (range(epochs)):
        total_loss=0
        count=0
        for ii, data_ in tqdm.tqdm(enumerate(data)):
            data_=torch.tensor(data_).long()
            x = data_.unsqueeze(1).to(device)
            optimizer.zero_grad()
            y = torch.zeros(x.shape).to(device).long()
            y[:-1], y[-1] = x[1:], x[0]
            output, _ = model(x)
            loss = criterion(output, y.view(-1))
            """
            hidden=None
            for k in range(2,max_lenth):
                data1=data_[:k]
                input_, target = data1[:-1, :], data1[1:, :]
                output, hidden = model(input_,hidden)
                loss = criterion(output, target.view(-1))
                optimizer.step()
            """
            loss.backward()
            optimizer.step()
            total_loss+=(loss.item())
            count+=1
        print(epoch,'loss=',total_loss/count)
        torch.save(model.state_dict(), 'model.bin' )
        chars=test(model)
        print(chars)


if __name__ == "__main__":
    # 获取数据
    data, word2ix, ix2word = get_data()
    with open("word2ix.pkl", 'wb') as outfile:
        pickle.dump(word2ix,outfile)
    with open("ix2word.pkl", 'wb') as outfile:
        pickle.dump(ix2word,outfile)

    data=data[:100]
    train()