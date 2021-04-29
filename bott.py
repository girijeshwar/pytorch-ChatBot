import numpy as np
import torch
import nltk
from nltk.stem.porter import  PorterStemmer


def tokenize(sentences):
    return nltk.word_tokenize(sentences)

def stemming(word):
    stemmer =PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence= [stemming(w) for w in tokenized_sentence]
    bag= np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    
    return bag



