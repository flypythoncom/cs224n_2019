import pickle

from flask import Flask,request
app = Flask(__name__)
import torch

from model import RNNModel,embed_size,hidden_dims,generate_poetry

with open("word2ix.pkl", 'rb') as outfile:
    word2ix=pickle.load(outfile)
with open("ix2word.pkl", 'rb') as outfile:
    ix2word=pickle.load(outfile)

device=torch.device('cpu')
model = RNNModel(len(word2ix), embed_size, hidden_dims)
init_checkpoint = 'model.bin'
model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/peom')
def predict():
    begin_word = request.args.get('text', '')
    ret=generate_poetry(model,word2ix,ix2word,device,begin_word)

    return ret

if __name__ == '__main__':
    app.run()