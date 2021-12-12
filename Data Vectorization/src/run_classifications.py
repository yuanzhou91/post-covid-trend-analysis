# Text Classification Example with Selected Corpus collected from Assignment1

# Author: Yuan Zhou
# Cited: Thomas W. Miller (2019-03-08)

# Compares text classification performance under random forests
# Three vectorization methods compared:
#     Analyst judgment
#     TfidfVectorizer from Scikit Learn
#     Doc2Vec from gensim

###############################################################################
### Note. Install all required packages prior to importing
###############################################################################
import multiprocessing
import sys

import re,string
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = True

from nltk.stem import PorterStemmer
#Functionality to turn stemming on or off
STEMMING = True  # judgment call, parsed documents more readable if False

MAX_NGRAM_LENGTH = 1  # try 1 for unigrams... 2 for bigrams... and so on
VECTOR_LENGTH = int(sys.argv[1]) # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False
SET_RANDOM = 9999


##############################
### Utility Functions 
##############################
# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references  
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text): 
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

##############################
### Prepare Training Data 
##############################
print('\n\n------------------------Prepare Training Data---------------------------------')

df = pd.read_json('items_with_grouped_labels.jl', lines=True)
classes_dict = {'Computers': 0, 'Health': 1, 'Business': 2, 'Other': 3}
classes_counts = [0, 0, 0, 0]

print('\nNumber of training documents per class:')
for target in df['topic']:
    idx = classes_dict[target]
    classes_counts[idx] = classes_counts[idx] + 1
    
for target_class in classes_dict.keys():
    print('Class: ', target_class, ', docs count:', classes_counts[classes_dict[target_class]])
    
X = df['body']
y = list(map(lambda x: classes_dict.get(x), df['topic']))

X_train, X_test, train_target, test_target = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=101)


train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF and Analyst judgment
for doc in X_train:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
print('Number of training documents:',
	len(train_text))	
print('Number of training token lists:',
	len(train_tokens))	

##############################
### Prepare Test Data 
##############################
test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF and Analyst judgment
for doc in X_test:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)
print('\nNumber of testing documents:',
	len(test_text))	
print('Number of testing token lists:',
	len(test_tokens))	

##############################
### Analyst judgment
##############################
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

word_counts_for_docs_train = list(map(lambda doc: word_count(doc), train_text))
word_counts_for_docs_test = list(map(lambda doc: word_count(doc), test_text))
word_count_whole_corpus = word_count(' '.join(train_text))

aj_keywords = []
aj_sorted_corpus_count_map = sorted(word_count_whole_corpus.items(), key=lambda item: item[1], reverse=True)
for term in {k: v for k, v in aj_sorted_corpus_count_map}:
    if len(aj_keywords) < VECTOR_LENGTH:
        aj_keywords.append(term)

aj_vectors_training = []
aj_vectors_test = []
for word_count in word_counts_for_docs_train:
    vector_for_doc = []
    for keyword in aj_keywords:
        vector_for_doc.append(word_count.get(keyword, 0))
    aj_vectors_training.append(vector_for_doc)

for word_count in word_counts_for_docs_test:
    vector_for_doc = []
    for keyword in aj_keywords:
        vector_for_doc.append(word_count.get(keyword, 0))
    aj_vectors_test.append(vector_for_doc)
    
aj_vectors_training = np.array(aj_vectors_training)
aj_vectors_test = np.array(aj_vectors_test)
print('\n\n---------------------Analyst judgment Vectorization------------------------------')
print('\nTraining aj_vectors_training.shape:', aj_vectors_training.shape)
print('\nTest aj_vectors_test.shape:', aj_vectors_test.shape)

aj_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
aj_clf.fit(aj_vectors_training, train_target)
aj_pred = aj_clf.predict(aj_vectors_test)  # evaluate on test set
print('\nAnalyst judgment/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, aj_pred, average='macro'), 3))


##############################
### TF-IDF Vectorization
##############################
print('\n\n---------------------TF-IDF Vectorization------------------------------')
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)
print('\nTFIDF vectorization. . .')
print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform
tfidf_vectors_test = tfidf_vectorizer.transform(test_text)
print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
tfidf_clf.fit(tfidf_vectors, train_target)
tfidf_pred = tfidf_clf.predict(tfidf_vectors_test)  # evaluate on test set
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))


##############################
### Compare features set
##############################
print('\n\n------------Compare features set: TF-IDF/Analyst Judgment----------------')

tfidf_keywords = set(tfidf_vectorizer.get_feature_names())
aj_keywords_set = set(aj_keywords)
#print('Most frequent keywords by TF-IDF: ', tfidf_keywords)
#print('Most frequent keywords by Analyst judgment: ', aj_keywords_set)

print('Unique keywords from only TF-IDF: ', tfidf_keywords.difference(aj_keywords_set))
print('Unique keywords from only Analyst judgment: ', aj_keywords_set.difference(tfidf_keywords))

def top_tfidf_feats(row, features, top_n=10):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df
def top_feats_in_doc(Xtr, features, row_id, top_n=10):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

print('Top `0 most important features in TF-IDF: \n',top_mean_feats(tfidf_vectors, tfidf_vectorizer.get_feature_names()))

print('Top `0 most important features in Analyst Judgment:')
print("{:^4}   {:^8}   {:^8}".format('rank', 'feature', 'count'))
idx = 0
for k, v in aj_sorted_corpus_count_map:
    print("{:^4}   {:^8}   {:^8}".format(idx, k, v))
    idx += 1
    if idx >= 10:
        break


###########################################
### Doc2Vec Vectorization
###########################################
# doc2vec paper:  https://cs.stanford.edu/~quocle/paragraph_vector.pdf
#     has a neural net with 1 hidden layer and 50 units/nodes
# documentation at https://radimrehurek.com/gensim/models/doc2vec.html
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
# tutorial on GitHub: 
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
print('\n\n---------------------Doc2Vec Vectorization------------------------------')
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_tokens)]
# print('train_corpus[:2]:', train_corpus[:2])

# Instantiate a Doc2Vec model with a vector size with 50 words 
# and iterating over the training corpus 40 times. 
# Set the minimum word count to 2 in order to discard words 
# with very few occurrences. 
# window (int, optional) – The maximum distance between the 
# current and predicted word within a sentence.
model_doc2vec = Doc2Vec(train_corpus, vector_size = VECTOR_LENGTH, window = 4, 
	min_count = 2, workers = cores, epochs = 10)

model_doc2vec.train(train_corpus, total_examples = model_doc2vec.corpus_count, 
	epochs = model_doc2vec.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_vectors = np.zeros((len(train_tokens), VECTOR_LENGTH)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_vectors[i,] = model_doc2vec.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_vectors.shape:', doc2vec_vectors.shape)
# print('doc2vec_vectors[:2]:', doc2vec_vectors[:2])

# vectorization for the test set
doc2vec_vectors_test = np.zeros((len(test_tokens), VECTOR_LENGTH)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_vectors_test[i,] = model_doc2vec.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_vectors_test.shape:', doc2vec_vectors_test.shape)

doc2vec_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_clf.fit(doc2vec_vectors, train_target) # fit model on training set
doc2vec_pred = doc2vec_clf.predict(doc2vec_vectors_test)  # evaluate on test set
print('\nDoc2Vec/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_pred, average='macro'), 3)) 


print('\n\n-------------------------------Summary-----------------------------------------')
print('\nVector Dimensions:', VECTOR_LENGTH),
print('\nAnalyst Judgment/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, aj_pred, average='macro'), 3))
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))
print('\nDoc2Vec/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_pred, average='macro'), 3)) 
print('\n------------------------------------------------------------------------')