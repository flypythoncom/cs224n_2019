import re
import tqdm
import torch
import collections
import numpy
from torch import nn

from model import RNNModel

device=

batch_size=128
embed_size=256

epochs=100
lr=0.001
def get_data():
    poetry_file = 'data/poetry.txt'
    special_character_removal = re.compile(r'[^\w。， ]', re.IGNORECASE)
    # 诗集
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8', ) as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = special_character_removal.sub('', content)
                content = content.replace(' ', '')
                if len(content) < 5:
                    continue
                if (len(content) > 12 * 6):
                    content_list = content.split("。")
                    for i in range(0, len(content_list), 2):
                        content_temp = '[' + content_list[i] + "。" + content_list[i + 1] + '。]'
                        content_temp = content_temp.replace("。。", "。")
                        poetrys.append(content_temp)
                else:
                    content = '[' + content + ']'
                    poetrys.append(content)
            except Exception as e:
                pass
    poetrys = sorted(poetrys, key=lambda line: len(line))
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
    ix2word = lambda word: word2ix.get(word, len(words))
    data = [list(map(ix2word, poetry)) for poetry in poetrys]
    data=numpy.array(data)
    return data,word2ix,ix2word

def train():

    # 获取数据
    data, word2ix, ix2word = get_data()
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=1)

    # 模型定义
    model = RNNModel(len(word2ix), batch_size, embed_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        for ii, data_ in tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()



    torch.save(model.state_dict(), 'model.bin' )
if __name__ == "__main__":

    train()