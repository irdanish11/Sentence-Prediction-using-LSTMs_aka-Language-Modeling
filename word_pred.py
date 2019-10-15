# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:20:58 2019

@author: danish
"""
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
import numpy as np
#This model just predict one word only
model = load_model('word_pred_Model4.h5')
tokenizer = load(open('tokenizer_Model4','rb'))
seq_len = 3 
def gen_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len,truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]
        
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

print('\n\n===>Enter --exit to exit from the program')
while True:
    seed_text  = input('Enter string: ')
    if seed_text.lower() == '--exit':
        break
    else:
        out = gen_text(model, tokenizer, seq_len=seq_len, seed_text=seed_text, num_gen_words=5)
        print('Output: '+seed_text+' '+out)
        
        
def get_prob(model, pad_encoded):
    prob = model.predict_proba(pad_encoded,verbose=0)
    max = 0
    max2 = 0
    max3 = 0
    ind3=1
    for i in range(len(prob[0])):
        if prob[0][i] > max:
            max = prob[0][i]
            ind1 = i
        if prob[0][i]>max2 and prob[0][i] <prob[0][np.argmax(prob)]:
            max2 = prob[0][i]
            ind2 = i
    for j in range(len(prob[0])):
        if prob[0][j]>max3 and prob[0][j] < max2:
            max3 = prob[0][j]
            ind3 = j
    #print(ind3)
    return [ind1,ind2,ind3], [prob[0][ind1],prob[0][ind2],prob[0][ind3]]

def gen_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = [seed_text,seed_text,seed_text,seed_text]
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len,truncating='pre')
        #pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]
        pred_word_ind1,pred_word_ind2,pred_word_ind3 = get_prob(model, pad_encoded)  
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

        
ind, prob = get_prob(model, pad_encoded)        
      
