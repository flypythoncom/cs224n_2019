import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size=128
embed_size=128
hidden_dims=256

def generate_poetry(model,word2ix,ix2word,device,begin,sent_len=4):
    start_idx=[word2ix['[']]
    end_word=''
    lens=0
    hidden = None
    ret=''
    data_ = torch.tensor([start_idx], device=device).long()
    output, hidden = model(data_, hidden)
    start_idx=[word2ix[begin]]
    ret+=begin
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

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)



    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()


        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(x)
        # output size: (seq_len,batch_size,hidden_dim)
        if hidden is None:
            output, hidden = self.lstm(embeds)
        else:
            h_0, c_0 = hidden
            output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden
