# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:23:04 2019

@author: Danish
"""

from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from pickle import dump



def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text

text = read_file('processed_sentences.txt')
tokens = text.split(" ")
tokens.pop(0)

train_len = 3+1
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
        


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 

#Collecting some information   
unique_words = tokenizer.index_word
unique_wordsApp = tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)

n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]


train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]

train_targets = to_categorical(train_targets, num_classes=vocabulary_size+1)
seq_len = train_inputs.shape[1]
train_inputs.shape



def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len,input_length=seq_len))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(vocabulary_size,activation='softmax'))
    opt_adam = optimizers.adam(lr=0.001)
    #You can simply pass 'adam' to optimizer in compile method. Default learning rate 0.001
    #But here we are using adam optimzer from optimizer class to change the LR.
    model.compile(loss='categorical_crossentropy',optimizer=opt_adam,metrics=['accuracy'])
    model.summary()
    return model

model = create_model(vocabulary_size+1,seq_len)
path = './checkpoints/word_pred_Model4.h5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(train_inputs,train_targets,batch_size=128,epochs=500,verbose=1,callbacks=[checkpoint])
model.save('word_pred_Model4.h5')
dump(tokenizer,open('tokenizer_Model4','wb'))    







