# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:47:21 2019

@author: danish
"""

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
import numpy as np
import pandas as pd

""" We input the sentence basically the model is only able to generate one word at each timestep when we input a string
    for example we input 'fan is not' the model will genrate the probabilities for all the words which are present in
    our vocabulary and then we choose the highest probability and the index of that probability will give us the word.
    Lets say 'working' word has the highest probability then we will pad this 'working' word into our orignal string
    and now we will have the string 'fan is not working' then this string will become the model so what we have to do
    we have to limit this string to 3 words, so what we do we remove the first word which is 'fan' and after that our
    string looks like 'is not working'. This process continues until we reach the limit of number of generated words
    in our case is 3. So when we generate three words in this way we stop predicting and display the final sentence as
    output. Now the output for our example will look like 'fan is not working in the'. We've successfuly generated the 
    3 words.
    
    Now our problem is to genrate multiple sentences as suggestions so our customer can choose from them, For example
    continuing the example lets say we want three sentences as suggestions so our model would have the something like
    this: 
        fan is not working in the
        fan is not operational in the
        fan is not operational please see
    For that what we have to do instead of choosing just a single word with the highest probability we choose three 
    different words with three successive highest probabilities. In this example we'll get three different words 
    for the input string 'fan is not' we will get 'working', 'operational' & 'fine'. We will store the probabilities 
    of all three words in a list. Then we will get three different sentences and after that for each of these sentence
    we will again gnerate three words now we will have total 9 different words. That would make 9 different sentences
    remember we only have generated two words for each sentence now for each of these 9 sentences we will again generate
    three different words and there respective probailities will be saved in a list and after that we will have 27 
    sentences. Now the challenge is we only have to suggest three sentences. So what we do we compute the conditional
    probailities for each of the sentnce from the list of probabilities that we have saved early on. Then we will
    return the three sentences with the highest conditional probabilities. 
    
    Note we cannot input a string greater or less than the defined sequence length because our LSTM model will only
    accept the specified sequence length otherwise it will generate error. So when the number of words is greater than 
    the specified sequence length then we remove the words from the start of the string so that its length becomes 
    equal to sequence length in our sequence length is 3. If number of words is less than the specified sequence length
    than we will pad the required number of some specific tokens in the start so that its length becomes equal to
    sequence length."""

model = load_model('word_pred_Model4.h5')
tokenizer = load(open('tokenizer_Model4','rb'))
seq_len = 3 
beam_width = 3

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
    #print(max, ind1)
    return [ind1,ind2,ind3], [prob[0][ind1],prob[0][ind2],prob[0][ind3]]

def gen_word(model, tokenizer, seq_len, seed_text, num_gen_words):
    input_text = seed_text

    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len,truncating='pre')
    ind, prob = get_prob(model,pad_encoded)
    return [tokenizer.index_word[ind[0]],tokenizer.index_word[ind[1]], tokenizer.index_word[ind[2]]], prob

def itr_fn(itr, word,exp2, ind, words_prob):
    y=[]
    n = []
    for k in range(beam_width**exp2):
            
        n.append(k)
        output_text[ind][0] = output_text[ind][0] + ' ' + word[itr-1]
        sent_prob[ind].append(words_prob[itr-1])
        ind += 1     
        if exp2==0:
            break 
    #print()
    y.append(n)
        #exp2 -= 1
    return y, ind


def list_gen(input_text):
    for text in range(beam_width**beam_width+1):
        output_text.append([input_text])
        sent_prob.append([])

def main_itr():
    exp = 1 #Start from power 1 and goes upto power equal to beam_width
    exp0 = 0 
    exp2 = 2 #Start from beam width-1
    exp3 = 3  
    for i in range(seq_len):
        #This loop will run equal to the number of words you want to generate
        #business of 2nd loop starts
        ind = 0
        word = []
        words_prob = []
        count = 0
        for itr in range(beam_width**exp0): 
            tm_word, prob = gen_word(model, tokenizer, seq_len, output_text[count][0], num_gen_words=3)
            count += beam_width**exp3 
            for a in range(len(tm_word)):
                word.append(tm_word[a])
                words_prob.append(prob[a])
        exp0 += 1
        for j in range(1, beam_width**exp+1):
            #business of third loop starts
            #print(a)
            y, ind = itr_fn(j, word, exp2, ind, words_prob)   
        if exp==beam_width:
            break
        exp += 1
        exp2 -= 1 
        exp3 -= 1
        
def cond_prob(a,b):
    #probability of b given a.
    return (a*b)/a        

        
def gen_sentence_sum(num_suggest_sent, return_list = False, return_df = False, print_out = True):
    """ Suggest the sentences on the basis of sum of all probabailities. This function print out the N number 
        of sentences which have the highest summed probability if used with the default arguments.
        
        num_suggest_sent: 
                    The number of sentences you want to generate
        return_list: 
                     By default it is False and if it set to True, then function will return a list of all the 
                     possible generated sentences.
        return_df: 
                   Default value is False but if set to True, it will return a Dataframe which will contain
                   all the generated sentences, conditional probabailities and probabiliti of each word in a
                   sentence.
        print_out: 
                   Default value is True, if False then do not prints the suggested sentences, and will return the
                   list of suggested sentences"""    
    sum_prob = []
    for i in range(len(sent_prob)):
        sum_prob.append(sum(sent_prob[i]))
    
    
    prob_sent = {}
    list_prob = {}
    for i in range(len(output_text)):
        prob_sent[sum_prob[i]] = output_text[i][0] 
        list_prob[sum_prob[i]] = sent_prob[i]
    
        
    sorted_sum_prob = sorted(prob_sent.keys())
    #Arange the probabilites in descending order
    sorted_sum_prob.reverse()
    sorted_prob = []
    sorted_sent = []
    for i in range(len(sorted_sum_prob)):
        sorted_prob.append(list_prob[sorted_sum_prob[i]])
        sorted_sent.append(prob_sent[sorted_sum_prob[i]])
        
    df = pd.DataFrame({'Sentences':sorted_sent, 'Sum of Probabilities':sorted_sum_prob, 'Probabilities':sorted_prob})
    sugested_sent = []
    for i in range(num_suggest_sent):
        if sugested_sent.append(df['Sentences'][i].replace('\n','')) not in sugested_sent:
            sugested_sent.append(df['Sentences'][i].replace('\n',''))  
        
    if return_list and return_df:
        return sugested_sent, df
    elif return_list and not return_df or not print_out:
        return sugested_sent
    elif not return_list and return_df:
        if print_out:
            for i in range(len(sugested_sent)):
                print(sugested_sent[i])
        return df
    else:
        for i in range(len(sugested_sent)):
            print(sugested_sent[i])

def gen_sentence_condProb(num_suggest_sent, return_list = False, return_df = False, print_out = True):
    """ Suggest the sentences on the basis of conditional probabaility. This function print out the N number 
        of sentences which have the highest conditional probability if used with the default arguments.
        
        num_suggest_sent: 
                    The number of sentences you want to generate
        return_list: 
                     By default it is False and if it set to True, then function will return a list of all the 
                     possible generated sentences.
        return_df: 
                   Default value is False but if set to True, it will return a Dataframe which will contain
                   all the generated sentences, conditional probabailities and probabiliti of each word in a
                   sentence.
        print_out: 
                   Default value is True, if False then do not prints the suggested sentences, and will return the
                   list of suggested sentences"""
                   
    keep_sent = 10 #Number of sentences that would be extracted from 1st set of conditional probabilities on the basis
                   #of highest probabilities. 
    first_set_prob = [] #list will hold the 1st set conditional probabilities
    second_set_prob = []
    for i in range(len(sent_prob)):
        #computes the conditional probablity as three words are generated for combination. So first set is computed by
        #the lets say if the first generated word for string (fan is not) is (working) then we generate the next three
        #possible words (lets say the next three words are in, the and please)and then we say compute the conditional
        #probabilities for all these three words given that the word (working) has occured.
        first_set_prob.append(cond_prob(a=sent_prob[i][0],b=sent_prob[i][1]))
        second_set_prob.append(0.0)
    
    df = pd.DataFrame({'Sentences':output_text, 'Cond Prob':first_set_prob, 'Probabilities':sent_prob})
    sorted_df = df.sort_values('Cond Prob', ascending=False).reset_index()
    
    for i in range(keep_sent):
        second_set_prob[i] = cond_prob(a=sorted_df['Probabilities'][i][1], b=sorted_df['Probabilities'][i][2])
    sorted_df['Cond Prob2'] = second_set_prob
    
    final_df = sorted_df.sort_values('Cond Prob2', ascending=False).reset_index()
    sugested_sent = []
    for i in range(num_suggest_sent):
        if final_df['Sentences'][i][0].replace('\n','') not in sugested_sent:
            sugested_sent.append(final_df['Sentences'][i][0].replace('\n',''))
    
    if return_list and return_df:
        return sugested_sent, final_df
    elif return_list and not return_df or not print_out:
        return sugested_sent
    elif not return_list and return_df:
        if print_out:
            for i in range(len(sugested_sent)):
                print(sugested_sent[i])
        return final_df
    else:
        for i in range(len(sugested_sent)):
            print(sugested_sent[i])
        
#To test single time
#input_text  = input('Enter string: ')
#output_text = []
#sent_prob = []
#list_gen(input_text)
#main_itr()
#dumy = output_text.pop(27)
#dumy = sent_prob.pop(27)
#df = gen_sentence_condProb(5)
   

print('\n\n===>Enter --exit to exit from the program')
while True:
    input_text  = input('Enter string: ')
    if input_text.lower() == '--exit':
        break
    else:
        output_text = []
        sent_prob = []
        list_gen(input_text)
        main_itr()
        dumy = output_text.pop(27)
        dumy = sent_prob.pop(27)
        df = gen_sentence_condProb(5, return_df=True)
    