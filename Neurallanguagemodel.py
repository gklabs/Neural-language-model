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
	for negative sample, randomly pick a word other than the word in the positive sample.

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
	for input_list in df.text:

		bigram_list=[]

		for i in range(len(input_list)-1):
			bigram_list.append((input_list[i], input_list[i+1]))
		print(bigram_list)
	return bigram_list

# returns a list that returns k negative sample bigrams for a given positive bigram from a dataset
def neg_sample_bigrammer(df):


def main():

    # print command line arguments
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    print(clean_train_stem.head(5))
    print(clean_train_nostem.head(5))

    print("creating positive bigrams")
    train_stem_pos_bigram= pos_sample_bigrammer(clean_train_stem)
    train_nostem_pos_bigram= pos_sample_bigrammer(clean_train_nostem)
    print("positive samples for training created")

    print("creating negative sample bigrams")
    neg_sample_bigrammer()


if __name__ == "__main__":
    main()
