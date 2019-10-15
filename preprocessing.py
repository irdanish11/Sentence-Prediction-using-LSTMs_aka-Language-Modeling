# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:23:04 2019

@author: danish
"""

import pandas as pd
import numpy as np
import re
import regex
import time

#reading the data from csv
data = pd.read_csv('EU-AU-Description-19-9-2019.csv', encoding="cp1252")

#removing the redundant lines
start_time  = time.time()
unique_data = []
for i in range(len(data)):
    if data['description'][i] not in unique_data:
        unique_data.append(data['description'][i])
        if i % 5000 == 0:
            print('{0}'.format(i)+' lines have been processed')
    else:
        None
print(start_time - time.clock)
end_time  = time.time()
print('Total time:', end_time - start_time)

# Method that will clean the data:
def clean_text(text):
    text = text.lower() #convert all the chracters into small letters
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=|.!?,]", "", text)
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text

sentence_count = {}
total_sentences = 0
for line in unique_data:
    if line not in sentence_count:
        sentence_count[line] = 1
    else:
        sentence_count[line] += 1
    total_sentences += 1
        
      
# Removing the lines that contains numeric data 
unique_data_str = []
for i in range(len(unique_data)):
    if type(unique_data[i]) is str:
        unique_data_str.append(unique_data[i])
    else:
        None

# Cleaning the data
clean_data = []
for text in unique_data_str:
    a = re.sub(r'[^a-zA-z ]+', '', text).strip()
    if len(a)>0:
        clean_data.append(clean_text(a))
    else:
        None

# Removing the lines which are to short or to long
short_data = []
for line in clean_data:
    if 2 <= len(line.split()) <= 25:
        short_data.append(line)
    else:
        None

# Counting the appearnce of each word in the corpus also calculates the number of unique words also
word2count = {}
total_words = 0
for text in short_data:
    for word in text.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
        total_words += 1
        
# creating a list that will only contain the words that appear more than 15 times
word15 = []
threshold = 15
for word, count in word2count.items():
    if count >= threshold:
        if len(word) > 1:
            word15.append(word)
            
# Removing the words from each string which appear less than 15 times
data_15 = []
for line in short_data:
    str1=''
    for word in line.split():
        if word in word15:
            str1 = " ".join((str1, word))
    data_15.append(str1)

#      
short_data_consize = []
for line in data_15:
    if 3 <= len(line.split()) <= 15:
        short_data_consize.append(line)
    else:
        None
        
clean_unique_data = []
for i in range(len(short_data_consize)):
    if short_data_consize[i] not in clean_unique_data:
        clean_unique_data.append(short_data_consize[i])
    else:
        None
        
# Total number of words in corpus after removing the words which appears less than 15 times and further cleaning
total_words_d15 = 0
for line in data_15:
    for word in line.split():
      total_words_d15 += 1 
""" Initially we had 437579 lines in our data after cleaning and preprocessing the data
    now our complete data will have 126003 lines, that means we removed 71.20% data which was useless"""   
    
#defining a function to save data
def write_txt(name, data):
    file1 = open("{0}.txt".format(name),"w") 
    for line in data:
        file1.writelines(line) 
        file1.writelines('\n') 
    file1.close() #to change file access modes

# Reading text file
#fl = open("EU-AU-Description-19-9-2019.txt","r+")  
#clean_unique_data = fl.read().splitlines()



# Splitting the cleaned and preprocessd data into 4 equal parts      
clean_unique_data_qtr1 = clean_unique_data[0:int(len(clean_unique_data)*0.25)]      
clean_unique_data_qtr2 = clean_unique_data[int(len(clean_unique_data)*0.25):int(len(clean_unique_data)*0.5)]  
clean_unique_data_qtr3 = clean_unique_data[int(len(clean_unique_data)*0.5):int(len(clean_unique_data)*0.75)]  
clean_unique_data_qtr4 = clean_unique_data[int(len(clean_unique_data)*0.75):len(clean_unique_data)]        

# writing data to text files
write_txt(name = 'EU-AU-Description-19-9-2019', data = clean_unique_data)
#write_txt(name = 'EU-AU-Description-19-9-2019_qtr1', data = clean_unique_data_qtr1)
#write_txt(name = 'job_base_EU_qtr2', data = clean_unique_data_qtr2)
#write_txt(name = 'job_base_EU_qtr3', data = clean_unique_data_qtr3)
#write_txt(name = 'job_base_EU_qtr4', data = clean_unique_data_qtr4)
