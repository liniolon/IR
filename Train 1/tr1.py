# -*- coding: utf-8 -*-
# Author: Amir Kouhkan

# Requirements libraries
import glob
import re
import math
import string

import pandas as pd


# Read all file and return them for processing
def get_all_files():
    f_2008 = glob.glob(r'OpinRankDataset\cars\2008\*')
    f_2009 = glob.glob(r'OpinRankDataset\cars\2009\*')
    f_2007 = glob.glob(r'OpinRankDataset\cars\2007\*')

    all_files = f_2007 + f_2008 + f_2009

    return all_files


# Extract content of <TEXT> tag from files with regex
def extract_text_data(data):
    all_text = list()

    for all_file in data:
        # print(all_file)
        file = open(all_file, 'r')
        read = file.read().strip()
        text = re.findall('<TEXT>(.*)</TEXT>', read)
        all_text.append(text)

    return all_text


# Extract content of <FAVORITE> tag from files with regex
def extract_fav_data(data):
    all_fav = list()

    for all_file in data:
        # print(all_file)
        file = open(all_file, 'r')
        read = file.read().strip()
        text = re.findall('<FAVORITE>(.*)</FAVORITE>', read)
        all_fav.append(text)

    return all_fav



def tokenizing_cleaning(data):
    read_stopwords = open('OpinRankDataset/stopwords.txt', 'r')
    stopwords = read_stopwords.read().strip()
    table = str.maketrans('', '', string.punctuation)

    text = list()

    for i in data:
        for j in i:
            text.append(j.split(' '))

    all_word = list()

    for i in text:
        for j in i:
            if j not in stopwords:
                all_word.append(j)

    stripped = [w.translate(table) for w in all_word]

    return stripped


def calculate_TF(words, word):
    word_of_tf = dict()
    word_count = len(word)

    for word, count in words.items():
        word_of_tf[word] = count / float(word_count)

    return word_of_tf


def calculate_IDF(list_of_docs):
    word_of_idf = dict()
    number_of_lists = len(list_of_docs)

    word_of_idf = dict.fromkeys(list_of_docs[0].keys(), 0)

    for lst in list_of_docs:
        for word, value in lst.items():
            if value > 0:
                word_of_idf[word] += 1

    for word, value in word_of_idf.items():
        word_of_idf[word] = math.log10(number_of_lists / float(value))

    return word_of_idf


def calculate_TFIDF(tfwords, idf):
    tf_idf = dict()

    print(tfwords)
    for word, value in tfwords.items():
        tf_idf[word] = value * idf[word]

    return tf_idf


all_word = tokenizing_cleaning(extract_text_data(get_all_files())) + tokenizing_cleaning(extract_fav_data(get_all_files()))

text_to_set = set(all_word)

word_dict1 = dict.fromkeys(text_to_set, 0)

for j in all_word:
    word_dict1[j] += 1

tf_1 = calculate_TF(word_dict1, all_word)
idf_1 = calculate_IDF([word_dict1])

print(calculate_TFIDF(tf_1, idf_1))

# pd.DataFrame([calculate_TFIDF(tf_1, idf_1)])
# print(all_word)