, BertTokenizerFast
import pickle

import torch
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
!pip
install
transformers
from transformers import BertTokenizer, BertForTokenClassification, AdamW, BertModel, AutoTokenizer, BertTokenizerFast
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import torch.nn as nn
import torch

import numpy as np
from google.colab import drive

drive.mount('/content/drive/')
% cd / content / drive / My
Drive / Colab
Notebooks / NER & POS

import pickle
import os

os.chdir("/content/drive/My Drive/Colab Notebooks/NER&POS")
!ls


def preprocess(d, is_test=0):
    res = []
    tokens = d['tokens']
    if not is_test:
        pos_tags = d['pos_tags']
        ner_tags = d['ner_tags']
    symbols = '{}()[]""\'\'``\\.,:;+-*/&|!...<>=~$'
    for sentence in tokens:
        res.append(sentence)
    if not is_test:
        return res, pos_tags, ner_tags
    return res


def get_num_tags(tags):
    t = []
    m = 0
    for sent in tags:
        for tag in sent:
            if tag not in t:
                t.append(tag)
                if tag > m:
                    m = tag
    print(t, len(t), m)
    return m


def align_labels(tokenized_input, labels):
    word_ids = tokenized_input.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels[word_idx])
            except:
                label_ids.append(-100)

        else:
            label_ids.append(labels[word_idx])
        previous_word_idx = word_idx

    return label_ids


class NERPOSDataset(Dataset):
    def __init__(self, sentences, ner_tags, pos_tags):
        self.sentences = sentences
        self.ner_tags = ner_tags
        self.pos_tags = pos_tags
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        ner_tag = self.ner_tags[idx]
        pos_tag = self.pos_tags[idx]
        sentence = " ".join(sentence)

        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors='pt'
        )
        '''
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        print(sentence)

        print(encoding)
        print(self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]))

        print(encoding.word_ids())
        print(ner_tag)
        '''
        ner_tag = align_labels(encoding, ner_tag)
        pos_tag = align_labels(encoding, pos_tag)
        # print(ner_tag)

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_tags': torch.tensor(ner_tag),
            'pos_tags': torch.tensor(pos_tag)
        }


f = open('train.pickle', 'rb')
train_data = pickle.load(f)
train_data, train_pos_tags, train_ner_tags = preprocess(train_data)
# f = open('processed_train.pkl', 'wb')
# pickle.dump(train_data, f)
# f = open('processed_train.pkl', 'rb')
# train_data = pickle.load(f)

f = open('validation.pickle', 'rb')
val_data = pickle.load(f)
val_data, val_pos_tags, val_ner_tags = preprocess(val_data)
# f = open('processed_validation.pkl', 'wb')
# pickle.dump(val_data, f)
# f = open('processed_validation.pkl', 'rb')
# val_data = pickle.load(f)

f = open('test.pickle', 'rb')
test_data = pickle.load(f)
test_data = preprocess(test_data, 1)
# f = open('processed_test.pkl', 'wb')
# pickle.dump(test_data, f)
##f = open('processed_test.pkl', 'rb')
# test_data = pickle.load(f)
'''''
maxlen = 200


'''
sentences = train_data  # List of tokenized sentences
tags = train_pos_tags  # List of corresponding POS tags
# num_tags=get_num_tags(train_ner_tags)


# Prepare your training data

for d in train_ner_tags:
    while len(d) < 128:
        d.append(-100)
    d = d[:128]
for d in train_pos_tags:
    while len(d) < 128:
        d.append(-100)
    d = d[:128]

dataset = NERPOSDataset(train_data[:50], train_ner_tags[:50], train_pos_tags[:50])

dataloader = DataLoader(dataset, batch_size=16)


class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        ### New layers:
        self.ner = nn.Linear(768, 9)
        self.pos = nn.Linear(768, 47)

    def forward(self, input_ids, attention_mask, training=False):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, return_dict=False)
        # print("LLLLLLLLLLLLLL")
        # print(sequence_output)
        # print(pooled_output)

        ner_output = self.ner(sequence_output)

        pos_output = self.pos(sequence_output)

        return ner_output, pos_output


# Load and prepare the BERT model for multitasking
model = CustomBERTModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ner_head=Dense(47,activation="softmax")
# pos_head=Dense(9,activation="softmax")

# model=Model(inputs=model,outputs=(ner_head,pos_head))
model.to(device)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
# Set up loss functions for NER and POS tagging
ner_loss_fn = torch.nn.CrossEntropyLoss()
pos_loss_fn = torch.nn.CrossEntropyLoss()

print(model)
# Training loop
epochs = 5
c = 0
for epoch in range(epochs):
    print(
        "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
    model.train()

    for batch in dataloader:
        c += 1
        print(c)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ner_tags = batch['ner_tags'].float()
        pos_tags = batch['pos_tags'].float()

        out_ner, out_pos = model(input_ids, attention_mask=attention_mask)

        out_ner = out_ner.cpu().detach().numpy()
        print(out_ner.shape)
        print(out_ner)
        out_ner = torch.Tensor(np.argmax(out_ner, axis=2))
        print(out_ner.shape)
        print(out_ner)
        print(ner_tags)

        out_pos = out_pos.cpu().detach().numpy()
        out_pos = torch.Tensor(np.argmax(out_pos, axis=2))

        # ner_loss = ner_loss_fn(ner_logits,ner_tags)
        # pos_loss = pos_loss_fn(pos_logits,pos_tags)
        loss1 = ner_loss_fn(out_ner, ner_tags)
        loss2 = pos_loss_fn(out_pos, pos_tags)
        loss1.requires_grad = True
        loss2.requires_grad = True
        loss = loss2 + loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
torch.save(model, 'fine-tuned_Bert4')

# model = torch.load('fine-tuned_Bert4')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model.to(device)


def predict_labels(sentence):
    sentence = " ".join(sentence)
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].squeeze().to(device)

    input_ids = input_ids.unsqueeze(0)
    attention_mask = encoding['attention_mask'].squeeze().to(device)
    attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        out_ner, out_pos = model(input_ids=input_ids, attention_mask=attention_mask)

    out_ner = out_ner.cpu().detach().numpy()
    out_ner = torch.Tensor(np.argmax(out_ner, axis=2))

    out_pos = out_pos.cpu().detach().numpy()
    out_pos = torch.Tensor(np.argmax(out_pos, axis=2))

    return out_ner, out_pos, encoding


ner_correct = 0
ner_all = 0
pos_correct = 0
pos_all = 0
for ind, data in enumerate(val_data[:10]):
    print(ind)
    ner_tags = val_ner_tags[ind]
    pos_tags = val_pos_tags[ind]
    while len(pos_tags) < 128:
        pos_tags.append(-100)
    while len(ner_tags) < 128:
        ner_tags.append(-100)
    ner_pred, pos_pred, enc = predict_labels(data)
    print("PPPPPPPPPPPPPPPPP")
    print(len(ner_tags), ner_tags)
    print(ner_pred.shape, ner_pred)

    ner_tags = align_labels(enc, ner_tags)
    print(len(ner_tags), ner_tags)

    pos_tags = align_labels(enc, pos_tags)

    for i, d in enumerate(ner_tags):
        if d != -100:
            ner_all += 1
            print(ner_pred[0][i], d)
            if ner_pred[0][i] == d:
                ner_correct += 1

    for i, d in enumerate(pos_tags):
        if d != -100:

            pos_all += 1

            print(pos_pred[0][i], d)

            if pos_pred[0][i] == d:
                pos_correct += 1

print(ner_correct / ner_all, pos_correct / pos_all)

test_pickle = {'tokens': test_data, 'ner_tags': [], 'pos_tags': []}
c = 0
for ind, data in enumerate(test_data):
    c += 1
    print(c)
    ner_pred, pos_pred = predict_labels(data)
    test_pickle['ner_tags'].append(ner_pred)
    test_pickle['pos_tags'].append(pos_pred)
test_pickle['tokens'] = np.array(test_pickle['tokens'], dtype=object)
test_pickle['ner_tags'] = np.array(test_pickle['ner_tags'], dtype=object)
test_pickle['pos_tags'] = np.array(test_pickle['pos_tags'], dtype=object)

f = open('test_pred.pkl', 'wb')
pickle.dump(test_pickle, f)

f = open('test_pred.pkl', 'rb')
test = pickle.load(f)
print(test['tokens'])
print(test['ner_tags'])
