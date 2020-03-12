#Neurallanguagemodel
'''
How to run:
File name + location of  data (tweet folder)
e.g. 
python3 Neurallanguagemodel.py /Users/gkbytes/nlm/tweet/train /Users/gkbytes/nlm/tweet/test
########################################
1. get data
	train
	validation
	test

2. clean
	Remove HTML tags
	Markup tags
	Lower case cap letters except stuff like USA
	No stop word removal
	tokenize at white space
	emoticon tokenizer (Tweet tokenizer)

3. bi-gram representation function for train and test
	2 negative sample for each positive sample
	for negative sample, 
		create the vocabulary of training sample
		randomly pick a word other than the word in the positive sample.

4. Feed forward Neural Network
	2 Hidden Layers of size 20
	initialize weights with random numbers
	LR= 0.00001/ tune
5. Predict and print accuracy in test

'''


# import dependencies
import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import math
import sys
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#pytorch related imports
import torch.nn as nn
import torch.optim as optim
import random


def get_data(path):
    pospath = path + '/positive'
    # create list to store text
    results = defaultdict(list)

    # loop through files and append text to list
    for file in Path(pospath).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results["text"].append(file_open.read())

    # read the list in as a dataframe
    df_pos = pd.DataFrame(results)

    #set directory path
    my_dir_path_neg = path + '/negative'

    # create list to store text
    results_neg = defaultdict(list)

    # loop through files and append text to list
    for file in Path(my_dir_path_neg).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results_neg["text"].append(file_open.read())
    # read the list in as a dataframe
    df_neg = pd.DataFrame(results_neg)
    df_neg.head()

    #add sentiment to both datasets and then combine them for test data 1 for positive and 0 for negative
    df_pos['Sentiment']=1
    df_neg['Sentiment']=0
    frames = [df_pos, df_neg]
    df = pd.concat(frames)

    # increase column width to see more of the tweets
    pd.set_option('max_colwidth', 140)

    # reshuffle the tweets to see both pos and neg in random order
    df = df.sample(frac=1).reset_index(drop=True)

    # explore top 5 rows
    df.head(5)
    return df



def clean(df):

    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext

    def removeat(text):
        atlist=[]
        for word in text:
            pattern = re.compile('^@')
            if re.match(pattern,word):
                #cleantext1 = re.sub(pattern, word[1:], word)
                atlist.append(word[1:])
            else:
                atlist.append(word)
        return atlist

    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist

    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))


    tokentweet=[]
    df.text= cleantweet
    for doc in df.text:
        tokentweet.append(TweetTokenizer().tokenize(doc))
    df.text= tokentweet

    removeattweet=[]
    for doc in df.text:
        removeattweet.append(removeat(doc))
    df.text =removeattweet

    lowertweet=[]
    for doc in df.text:
        lowertweet.append(tolower(doc))
    df.text = lowertweet

    tweets=[]
    for x in df.text:
        tweet = ''
        for word in x:
            tweet += word+' '
        tweets.append(word_tokenize(tweet))
    df.text= tweets

    #stemming
    stemtweets=[]
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    #ps= PorterStemmer()
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    df_unstemmed = pd.DataFrame()
    df_unstemmed['text'] = df['text']
    df_unstemmed['Sentiment'] = df['Sentiment']
    df_stemmed = pd.DataFrame()
    df_stemmed['text'] = df['stemmed']
    df_stemmed['Sentiment'] = df['Sentiment']
    
    ### Finalize both the stemmed and unstemmed dataframes
    #df_unstemmed = df.drop(['stemmed'], axis=1)
    #df_unstemmed.head()

    # create a df with stemmed text
    #df_stemmed = df.drop(['text'], axis=1)
    
    return df_stemmed,df_unstemmed

#returns a list containing bigrams
def pos_sample_bigrammer(df):
	bigram_list=[]
	for input_list in df.text:

		for i in range(len(input_list)-1):
			bigram_list.append((input_list[i], input_list[i+1]))
		#print(bigram_list)
	return bigram_list


#creating vocabulary
def createvocab(df):
    V=[]
    for tweet in df.text:
        for keyword in tweet:
            if keyword  in V:
                continue
            else :
                V.append(keyword)
    return V


# returns a list that returns k negative sample bigrams for a list of given positive bigrams
def neg_sample_bigrammer(bigramlist,vocab,k):
	end= len(vocab)
	negsample=[]
	for posbigram in bigramlist:
		word=str(posbigram[1])
		if word in vocab:
			num= vocab.index(word)
			neglist= random.sample([i for i in range(0,end) if i not in [num]],k)
			for j in neglist:
				negsample.append((posbigram[0],vocab[j]))
	return negsample

def NeuralNetwork():
	

def main():

    # print command line arguments
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    

    print("creating the vocabulary for stemmed and unstemmed data")
    Vocab_stem = createvocab(clean_train_stem)
    Vocab_nostem = createvocab(clean_train_nostem)
    print("vocabulary created")
    print("Stemmed vocabulary length=",len(Vocab_stem))
    print("No stem vocabulary length=",len(Vocab_nostem))

    print("creating positive bigrams")
    train_stem_pos_bigram= pos_sample_bigrammer(clean_train_stem)
    train_nostem_pos_bigram= pos_sample_bigrammer(clean_train_nostem)
    print("positive samples for training created")
    print("No of no stem pos bigrams=", len(train_nostem_pos_bigram))
    print("No of stem pos bigrams=" ,len(train_stem_pos_bigram))

    print("creating negative sample bigrams")
    train_stem_neg_bigram = neg_sample_bigrammer(train_stem_pos_bigram, Vocab_stem,2)
    train_nostem_neg_bigram = neg_sample_bigrammer(train_nostem_pos_bigram, Vocab_nostem,2)
    print("negative samples created")
    
    print("No of no stem neg bigrams=", len(train_nostem_neg_bigram))
    print("No of stem neg bigrams=" ,len(train_stem_neg_bigram))

    #create a training dataframe with positive and negative samples and adding 1,0  for them
    train_stem_data=pd.DataFrame()
    train_nostem_data=pd.DataFrame()

    train_stem_data['bigram'] = train_stem_pos_bigram + train_stem_neg_bigram
    train_nostem_data['bigram']= train_nostem_pos_bigram + train_nostem_neg_bigram
    
    y_nostem =[0]*len(train_nostem_pos_bigram) + [1]*len(train_nostem_neg_bigram)
    y_stem= [0]*len(train_stem_pos_bigram)+ [1]*len(train_stem_neg_bigram)

    train_stem_data['labels'] = y_stem
    train_nostem_data['labels'] = y_nostem

    print("train data is ready for stem and no stem ")



    #create a neural language model




if __name__ == "__main__":
    main()
