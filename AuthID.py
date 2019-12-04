#HSS 322 Project - AuthID
#v3.0
#Siddharth Chaini (17275) and Siddharth Bachoti (17274)
#https://github.com/AKnightWing/AuthID
#4/12/19

import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import operator
import time

def slowprint(words):
    for char in words:
        time.sleep(0.02)
        print(char, end='', flush=True)

if os.name=="posix":
	color1='\u001b[36;1m'
	color2='\033[0m'
	color3='\u001b[32;1m'
else:
	color1=''
	color2=''
	color3=''


if os.name=="posix":
	os.system('clear')
slowprint("Hello. Welcome to " + color1 + 'AuthID '+color2 +". This software hopes to identify an unknown author of a given text.\n")
time.sleep(1)
slowprint("This is done by analysing the charactertistic ngram frequencies of the authors' works in the training set, and the matched to data in the test set.\n")
time.sleep(1)
slowprint("For more info, please read the readme and the project report.\n\n")
time.sleep(2)

print("Please have the train and test data directories in the proper place as mentioned in the readme.\n\n")
py_path=os.path.dirname(os.path.abspath(__file__))

def normalize_counter(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return Counter({key:value*factor for key,value in d.items()})

def normalize_dict(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}


traindatapath=os.path.join(py_path,"Train Data")
os.chdir(traindatapath)
train_authors=os.listdir(traindatapath)

author_dict={}
bigram_author_dict={}
trigram_author_dict={}

for author in train_authors:
    all_text_by_author=""
    os.chdir(os.path.join(traindatapath,author))
    all_works=os.listdir()
    print("******************")
    print("Analysing all works by {}".format(author))
    print("-------")
    for work in all_works:
        print("Read '{}' by '{}' succesfully.".format(work.title(),author))
        file=open(work,encoding="utf8")
        text=file.read()
        all_text_by_author=all_text_by_author+text
        file.close()
    token=word_tokenize(all_text_by_author)
    
    c=normalize_counter(Counter(token))
    author_dict[author]=dict(c)
    
    bigram = nltk.bigrams(token)
    c2=normalize_counter(Counter(bigram))
    bigram_author_dict[author]=dict(c2)
    
    trigram = nltk.trigrams(token)
    c3=normalize_counter(Counter(trigram))
    trigram_author_dict[author]=dict(c3)
    
    os.chdir(traindatapath)


print("******************")
print("List of all authors whose works have been trained are:")
for author in train_authors:
    print("\t"+str(author))


testdatapath=os.path.join(py_path,"Test Data")

os.chdir(testdatapath)
test_cases=os.listdir()


print("****RESULTS****")
for case in test_cases:
    book_name=case.split(".txt")[0]
    f1=open(case,encoding="utf8")
    text=f1.read()
    token_words=word_tokenize(text)

    c1=normalize_counter(Counter(token_words))
    d1=dict(c1.most_common(len(c1)))

    bigram_test = nltk.bigrams(token_words)
    c2=normalize_counter(Counter(bigram_test))
    d2=dict(c2.most_common(len(c2)))

    trigram_test = nltk.trigrams(token_words)
    c3=normalize_counter(Counter(trigram_test))
    d3=dict(c3.most_common(len(c3)))

    error_author_dict={}
    bigram_error_author_dict={}
    trigram_error_author_dict={}

    for author in train_authors:
        #UnigramModel
        current_author_dict=author_dict[author]
        error=0
        max_uni_error=0
        for key in d1:
            max_uni_error=max_uni_error+abs(d1[key])
            if key in current_author_dict:
                error=error+abs(d1[key]-current_author_dict[key])  #Or try removing **2
            else:
                error=error+abs(d1[key])        
        error_author_dict[author]=error


        #BigramModel
        bigram_current_author_dict=bigram_author_dict[author]
        bigram_error=0
        max_bi_error=0
        for key in d2:
            max_bi_error=max_bi_error+abs(d2[key])
            if key in bigram_current_author_dict:
                bigram_error=bigram_error+abs(d2[key]-bigram_current_author_dict[key])
            else:
                bigram_error=bigram_error+abs(d2[key])
        bigram_error_author_dict[author]=bigram_error

        #TrigramModel
        trigram_current_author_dict=trigram_author_dict[author]
        trigram_error=0
        max_tri_error=0
        for key in d3:
            max_tri_error=max_tri_error+abs(d3[key])
            if key in trigram_current_author_dict:
                trigram_error=trigram_error+abs(d3[key]-trigram_current_author_dict[key])
            else:
                trigram_error=trigram_error+abs(d3[key])
        trigram_error_author_dict[author]=trigram_error

    prob_author_dict={}
    prob_bigram_author_dict={}
    prob_trigram_author_dict={}

    master_prob_dict={}
    #Interpolation
    for key in error_author_dict:
        prob_author_dict[key]=(max_uni_error-error_author_dict[key])/max_uni_error

    for key in bigram_error_author_dict:
        prob_bigram_author_dict[key]=(max_bi_error-bigram_error_author_dict[key])/max_bi_error

    for key in trigram_error_author_dict:
        prob_trigram_author_dict[key]=(max_tri_error-trigram_error_author_dict[key])/max_tri_error


    for key in error_author_dict:
        master_prob_dict[key]=(0.99*prob_author_dict[key])+(0.005*prob_bigram_author_dict[key])+(0.005*prob_trigram_author_dict[key])*100


    key_min = min(error_author_dict.keys(), key=(lambda k: error_author_dict[k]))
    bigram_key_min = min(bigram_error_author_dict.keys(), key=(lambda k: bigram_error_author_dict[k]))
    trigram_key_min = min(trigram_error_author_dict.keys(), key=(lambda k: trigram_error_author_dict[k]))
    master_prob_max_key = max(master_prob_dict.keys(), key=(lambda k: master_prob_dict[k]))
    print("The book '{}' is predicted to have been authored by '{}'".format(book_name.title(),master_prob_max_key.title()))