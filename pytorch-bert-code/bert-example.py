#https://github.com/huggingface/transformers

#https://huggingface.co/transformers/quickstart.html
#BERT example

#pip install transformers
#老的pytorch_transformers

import torch
import torch.nn as nn
from transformers  import BertConfig, BertModel
from transformers.tokenization_bert  import BertTokenizer as tokenization
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#get_bert_model
# bert预训练模型：
# pytorch_model.bin
# config.json
# vocab.txt
bert_path = './bert'
do_lower_case=True

bert_config_file = os.path.join(bert_path, f'bert_config.json')
vocab_file = os.path.join(bert_path, f'vocab.txt')
init_checkpoint = os.path.join(bert_path, f'pytorch_model.bin')

#加载配置
bert_config = BertConfig.from_json_file(bert_config_file)

# 加载词典
tokenizer = tokenization(vocab_file=vocab_file, do_lower_case=do_lower_case)

# 加载模型
model_bert =  BertModel.from_pretrained(bert_path)
model_bert.to(device)


# Tokenize input
text = "乌兹别克斯坦议会立法院主席获连任"
tokenized_text = tokenizer.tokenize(text)
tokenized_text=['[CLS]'] + tokenized_text + ['[SEP]']
# Convert token to vocabulary indices
# input_ids：一个形状为[batch_size, sequence_length]的torch.LongTensor，在词汇表中包含单词的token索引
input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segment_ids ：形状[batch_size, sequence_length]的可选torch.LongTensor，在[0, 1]中选择token类型索引。类型0对应于句子A，类型1对应于句子B。
segment_ids = [0]*len(input_ids)
# input_mask：一个可选的torch.LongTensor，形状为[batch_size, sequence_length]，索引在[0, 1]中选择。
input_mask = [1]*len(input_ids)

# Convert inputs to PyTorch tensors
input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
print("input_ids",input_ids.size())
input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)# attention_mask,可以不输入
segments_tensors = torch.tensor([segment_ids], dtype=torch.long).to(device)
#输出
all_encoder_layer, pooled_output = model_bert(input_ids,input_mask,token_type_ids=segments_tensors)
# all_encoder_layers：一个大小为[batch_size, sequence_length，hidden_size]的torch.FloatTensor列表，
# 它是每个注意块末端隐藏状态的完整序列列表（即BERT-base的12个完整序列，BERT-large的24个完整序列）
# pooled_output：一个大小为[batch_size, hidden_size] 的torch.FloatTensor，
# 它是在与输入（CLF）的第一个字符相关联的隐藏状态之上预训练的分类器的输出，用于训练Next - Sentence任务（参见BERT的论文）

#如果我们要输出embeding 表示，只使用all_encoder_layer
print('all_encoder_layer',all_encoder_layer.shape)
print('pooled_output',pooled_output.size())
#如果要分类，使用pooled_output

#padding
max_seq_length=300

text = "乌兹别克斯坦议会立法院主席获连任"
tokenized_text = tokenizer.tokenize(text)
tokenized_text=['[CLS]'] + tokenized_text + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
input_mask = [1]*len(input_ids)

padding = [0] * (max_seq_length - len(input_ids))
input_ids += padding
input_mask += padding
input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
print("padding input_ids",input_ids.size())

model_bert.eval()
with torch.no_grad():
    all_encoder_layer, pooled_output = model_bert(input_ids,attention_mask= input_mask)
    print('padding all_encoder_layer', all_encoder_layer.shape)
    print('padding pooled_output', pooled_output.size())