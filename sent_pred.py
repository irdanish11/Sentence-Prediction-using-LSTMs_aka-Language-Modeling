# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:10:58 2020

@author: Danis
"""


from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
import numpy as np
import warnings


def gen_list_words(input_text, model, tokenizer, num_words, seq_len=3):
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    #padding
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    #getting probabilities
    prob = model.predict_proba(pad_encoded,verbose=0)
    prob_lst = list(prob[0])
    pred_words = []
    indicies = []
    prob_pred = []
    for i in range(num_words):
        ind = np.argmax(prob_lst)
        indicies.append(ind)
        pred_words.append(tokenizer.index_word[ind])
        prob_pred.append(prob_lst[ind])
        prob_lst[ind] = 0
    return pred_words, prob_pred, indicies

def get_dic(seed_text, num_words):
    sent = {}
    probs = {}
    for i in range(num_words**num_words):
        sent[i] = seed_text.split(' ')
        probs[i] = []
    return sent, probs

def fetch_dic(input_text, num_words):
    pwr = num_words - 1  
    sent, probs = get_dic(input_text, num_words)  
    fetch_ind = {}
    for i in range(num_words):
        fetch_ind[i] = []
        #calculating number of sequences
        nbr_seq = num_words**pwr
        pwr -= 1
        min_range = 0
        max_range = nbr_seq
        for j in range(0, num_words**(i+1)):
            fetch_ind[i].append(min_range)
            min_range = max_range
            max_range += nbr_seq
    return fetch_ind, sent, probs

def get_sentences(input_text, num_words, model, tokenizer):
    fetch_ind, sent, probs = fetch_dic(input_text, num_words)
    pwr = num_words - 1  
    #sent, probs = get_dic(seed_text)  
    for i in range(num_words):
        #calculating number of sequences
        nbr_seq = num_words**pwr
        pwr -= 1
        min_range = 0
        max_range = nbr_seq
        
        for j in range(0, num_words**(i+1)):
            input_text = ' '.join(sent[fetch_ind[i][j]])
            #Getting predictions of words
            pred_words, prob_pred, indicies = gen_list_words(input_text, model, tokenizer, 
                                                              num_words, seq_len=3)
            for k in range(min_range, max_range):
                #
                n = fetch_ind[i].index(min_range)%num_words
                sent[k].append(pred_words[n])
                probs[k].append(prob_pred[n])
            #print(min_range)
            min_range = max_range
            max_range += nbr_seq
    return sent, probs

def get_sorted_sent(sent, probs):
    lst = list(probs.values())
    probs_lst = []
    for prob in lst:
        probs_lst.append(np.prod(prob))
    
    sorted_sent = []
    tmp = probs_lst.copy()
    for i in range(len(tmp)):
        idx = np.argmax(tmp)
        sing_sent = ' '.join(sent[idx])
        if '\n' in sing_sent:
            sing_sent = sing_sent.replace('\n', '')
        sorted_sent.append(sing_sent)
        tmp[idx] = 0
    new = []
    new_probs_lst = []
    for i, s in enumerate(sorted_sent):
        if s not in new:
            new.append(s)
            new_probs_lst.append(probs_lst[i])
    sorted_sent = new
    return sorted_sent	, new_probs_lst

def sort_on_list(lst1, lst2):
    tmp = lst2.copy()
    
    lst3 = []
    lst4 = []
    for i in range(len(tmp)):
        idx = np.argmax(tmp)
        lst3.append(lst1[idx])
        lst4.append(tmp[idx])
        tmp[idx] = 0
    return lst3, lst4
    
def compare_sent(input_text, model, tokenizer, num_words):
    sorted_sent = []
    probs_lst = []
    if type(num_words) != list:
        sent, probs = get_sentences(input_text, num_words, model, tokenizer)
        sorted_sent, probs_lst = get_sorted_sent(sent, probs)
    elif type(num_words) == list:
        for n_w in num_words:
            sent, probs = get_sentences(input_text, n_w, model, tokenizer)
            sent_sort, lst_prob = get_sorted_sent(sent, probs)
            sorted_sent.extend(sent_sort)
            probs_lst.extend(lst_prob)
        sent, prob = sort_on_list(sorted_sent, probs_lst)
    return sent, prob

def predict_sent(input_text, model, tokenizer, num_words=3, num_sent=3):
    sorted_sent, probs_lst = compare_sent(input_text, model, tokenizer, num_words)
    l = len(sorted_sent)
    if num_sent > l:
        s1 = 'Value given to `num_sent` argument is greater than total number of generated sentences.'
        s2 = '\nValue given to argument `num_words`: {0}, Total number of generated sentences: {1}.'.format(num_sent, l)
        s3 = '\nDefault values of sentences will returned'
        warnings.warn(s1+s2+s3, RuntimeWarning)
        num_sent = l
    out_sent = sorted_sent[0:num_sent]
    return out_sent, probs_lst

########## Run following lines on global scope ########## 
num_words = 3
model = load_model('word_pred_Model4.h5')
tokenizer = load(open('tokenizer_Model4','rb'))

#Making predictions
input_text = 'ac is not' 
out_sent, probs = predict_sent(input_text, model, tokenizer, num_words=[4], num_sent=3)
    


