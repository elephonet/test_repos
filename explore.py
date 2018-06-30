
import numpy as np
import pandas as pd
import re
import jieba
import jieba.posseg as pseg

# import codecs
from gensim import corpora, models, similarities

r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
stop_word = ['、','。','“','”','《','》','！','，','：','；','？']


df = pd.read_csv('base_add_data.csv',encoding='gb18030')
df['经营范围'] = df['经营范围'].map(lambda x: re.sub('\（(.*?)\）','',str(x)))
df1=df.sample(n=100)
print (df1['经营范围'].head())

#df_enterprice = pd.read_csv('enterprice_cluster_corpus.txt',encoding='gb18030').dropna()
#print (df_enterprice.head())
#df['range_cut'] = df['经营范围'].map(lambda x: list(jieba.cut(x)))
#print (df.range_cut.head())
#df['range_cut'] = df['range_cut'].map(lambda x: set([i for i in x if i not in stop_word]))

#print (df.range_cut.head())

def enter_cluster_corpus(filename):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v']
    f = open(filename,encoding='utf-8') # 'enterprice_cluster_corpus.txt'
    corpus = []
    for line in f.readlines():
        result = []
        if line is not None:
            words = pseg.cut(line)
        for word, flag in words:
            if flag not in stop_flag:
                result.append(word)
        corpus.append(list(set(result)))
    return corpus



def tokenization(filename):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v']
    corpus = []
    with open(filename, 'r',encoding='gb18030') as f:
        #text = f.read()
        for line in f.readlines():
            result = []
            words = pseg.cut(line)
            for word, flag in words:
                if flag not in stop_flag :
                    result.append(word)
            corpus.append(result)
    return corpus

def _tokenization(df):
    corpus = []
    text = ''
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r','v']
    for i in range(len(df)):
        result = []
        text = df.ix[i,'经营范围']
        words = pseg.cut(text)
        for word, flag in words:
            if flag not in stop_flag:
                result.append(word)
        #print (result)
        corpus.append(result)
    return corpus


if __name__ == '__main__':
    enterprice = enter_cluster_corpus('enterprice_cluster_corpus.txt')
    #print (enterprice)
    corpus = _tokenization(df)
    #print (len(corpus))
    corpus_add = []
    corpus_add.extend(corpus)
    corpus_add.extend(enterprice)
    #print (len(corpus_add))

    dictionary = corpora.Dictionary(corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    print (dictionary)
    print (doc_vectors[:3])

    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    #print (len(tfidf_vectors))
    print (tfidf_vectors[0])

    #query = corpus[0]
    #query_bow = dictionary.doc2bow(query)
    #query_bow1 = tfidf_vectors[0]
    #print (dictionary[9539])
    print (doc_vectors[10053])

    query2 = tokenization('enterprice_cluster.txt')
    query_bow2 = dictionary.doc2bow(query2[2])
    query_bow2 = tfidf[query_bow2]
    #print (query2[1])

    index = similarities.MatrixSimilarity(tfidf_vectors)

    from operator import itemgetter

    sims = index[query_bow2]
    sort = sorted(list(enumerate(sims)), key=itemgetter(1, 0), reverse=1)
    #print (sort)

