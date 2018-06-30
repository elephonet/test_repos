#-*- coding:utf-8

'''

preprocess.py
这个文件的作用是做文档预处理，
讲每篇文档，生成相应的token_list
只需执行最后documents_pre_process函数即可。

'''

import nltk
import traceback
import jieba
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from collections import defaultdict

# 分词 - 英文
def tokenize(document):
    try:

        token_list = nltk.word_tokenize(document)

        #print "[INFO]: tokenize is finished!"
        return token_list

    except Exception,e:
        print traceback.print_exc()

# 分词 - 中文
def tokenize_chinese(document):
    try:

        token_list = jieba.cut( document, cut_all=False )

        #print "[INFO]: tokenize_chinese is finished!"
        return token_list

    except Exception,e:
        print traceback.print_exc()

# 去除停用词
def filtered_stopwords(token_list):
    try:


        token_list_without_stopwords = [ word for word in token_list
                                         if word not in stopwords.words("english")]


        #print "[INFO]: filtered_words is finished!"
        return token_list_without_stopwords
    except Exception,e:
        print traceback.print_exc()

# 去除标点
def filtered_punctuations(token_list):
    try:
        punctuations = ['', '\n', '\t', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        token_list_without_punctuations = [word for word in token_list
                                                         if word not in punctuations]
        #print "[INFO]: filtered_punctuations is finished!"
        return token_list_without_punctuations

    except Exception,e:
        print traceback.print_exc()

# 词干化
def stemming( filterd_token_list ):
    try:

        st = LancasterStemmer()
        stemming_token_list = [ st.stem(word) for word in filterd_token_list ]

        #print "[INFO]: stemming is finished"
        return stemming_token_list

    except Exception,e:
        print traceback.print_exc()

# 去除低频单词
def low_frequence_filter( token_list ):
    try:

        word_counter = defaultdict(int)
        for word in token_list:
            word_counter[word] += 1

        threshold = 0
        token_list_without_low_frequence = [ word
                                             for word in token_list
                                             if word_counter[word] > threshold]

        #print "[INFO]: low_frequence_filter is finished!"
        return token_list_without_low_frequence
    except Exception,e:
        print traceback.print_exc()

"""
功能：预处理
@ document: 文档
@ token_list: 预处理之后文档对应的单词列表
"""
def pre_process( document ):
    try:

        token_list = tokenize(document)
        #token_list = filtered_stopwords(token_list)
        token_list = filtered_punctuations(token_list)
        #token_list = stemming(token_list)
        #token_list = low_frequence_filter(token_list)

        #print "[INFO]: pre_process is finished!"
        return token_list

    except Exception,e:
        print traceback.print_exc()

"""
功能：预处理
@ document: 文档集合
@ token_list: 预处理之后文档集合对应的单词列表
"""
def documents_pre_process( documents ):
    try:

        documents_token_list = []
        for document in documents:
            token_list = pre_process(document)
            documents_token_list.append(token_list)

        print "[INFO]:documents_pre_process is finished!"
        return documents_token_list

    except Exception,e:
        print traceback.print_exc()

#-----------------------------------------------------------------------
def test_pre_process():

    documents = ["he,he,he,we are happy!",
                 "he,he,we are happy!",
                 "you work!"]
    documents_token_list = []
    for document in documents:
        token_list = pre_process(document)
        documents_token_list.append(token_list)

    for token_list in documents_token_list:
        print token_list

#test_pre_process()

#-*- coding:utf-8

'''

lda_model.py
这个文件的作用是lda模型的训练
根据预处理的结果，训练lda模型

'''

from pre_process import documents_pre_process
from gensim import corpora, models, similarities
import traceback

# 训练tf_idf模型
def tf_idf_trainning(documents_token_list):
    try:

        # 将所有文章的token_list映射为 vsm空间
        dictionary = corpora.Dictionary(documents_token_list)

        # 每篇document在vsm上的tf表示
        corpus_tf = [ dictionary.doc2bow(token_list) for token_list in documents_token_list ]

        # 用corpus_tf作为特征，训练tf_idf_model
        tf_idf_model = models.TfidfModel(corpus_tf)

        # 每篇document在vsm上的tf-idf表示
        corpus_tfidf = tf_idf_model[corpus_tf]

        #print "[INFO]: tf_idf_trainning is finished!"
        return dictionary, corpus_tf, corpus_tfidf

    except Exception,e:
        print traceback.print_exc()

# 训练lda模型
def lda_trainning( dictionary, corpus_tfidf, K ):
    try:

        # 用corpus_tfidf作为特征，训练lda_model
        lda_model = models.LdaModel( corpus_tfidf, id2word=dictionary, num_topics = K )

        # 每篇document在K维空间上表示
        corpus_lda = lda_model[corpus_tfidf]

        #print "[INFO]: lda_trainning is finished!"
        return lda_model, corpus_lda

    except Exception,e:
        print traceback.print_exc()

'''
功能:根据文档来训练一个lda模型，以及文档的lda表示
    训练lda模型的用处是来了query之后，用lda模型将queru映射为query_lda
@documents:原始文档raw material
@K:number of topics
@lda_model:训练之后的lda_model
@corpus_lda:语料的lda表示
'''
def get_lda_model( documents, K ):
    try:

        # 文档预处理
        documents_token_list = documents_pre_process( documents )

        # 获取文档的字典vsm空间,文档vsm_tf表示,文档vsm_tfidf表示
        dict, corpus_tf, corpus_tfidf = tf_idf_trainning( documents_token_list)

        # 获取lda模型,以及文档vsm_lda表示
        lda_model, corpus_lda = lda_trainning( dict, corpus_tfidf, K )

        print "[INFO]:get_lda_model is finished!"
        return lda_model, corpus_lda, dict, corpus_tf, corpus_tfidf

    except Exception,e:
        print traceback.print_exc()


#-*- coding:utf-8

'''
similarity.py
这个文件的作用是训练后的的lda模型，对语料进行相似度的计算

'''

from gensim import corpora, models, similarities
import traceback

'''
这个函数没有用到
'''
# 基于lda模型的相似度计算
def lda_similarity( query_token_list, dictionary, corpus_tf, lda_model ):
    try:

        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )

        # 在dictionary建立query的vsm_tf表示
        query_bow = dictionary.doc2bow( query_token_list )

        # 查询在K维空间的表示
        query_lda = lda_model[query_bow]

        # 计算相似度
        # simi保存的是 query_lda和corpus_lda的相似度
        simi = index[query_lda]
        query_simi_list = [ item for _, item in enumerate(simi) ]
        return query_simi_list

    except Exception,e:
        print traceback.print_exc()

'''
功能：语聊基于lda模型的相似度计算
@ corpus_tf:语聊的vsm_tf表示
@ lda_model:训练好的lda模型
'''
def lda_similarity_corpus( corpus_tf, lda_model ):
    try:

        # 语料库相似度矩阵
        lda_similarity_matrix = []

        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )

        # 计算相似度
        for query_bow in corpus_tf:

            # K维空间表示
            query_lda = lda_model[query_bow]

            # 计算相似度
            simi = index[query_lda]

            # 保存
            query_simi_list = [item for _, item in enumerate(simi)]
            lda_similarity_matrix.append(query_simi_list)

        print "[INFO]:lda_similarity_corpus is finished!"
        return lda_similarity_matrix

    except Exception,e:
        print traceback.print_exc()


#-*- coding:utf-8
'''
save_result.py
这个文件的作用是保存结果
'''


import traceback

def save_similarity_matrix(matrix, output_path):
    try:

        outfile = open( output_path, "w" )

        for row_list in matrix:
            line = ""
            for value in row_list:
                line += ( str(value) + ',' )
            outfile.write(line + '\n')

        outfile.close()
        print "[INFO]:save_similarity_matrix is finished!"
    except Exception,e:
        print traceback.print_exc()

#-*- coding:utf-8

'''
test_lda_main.py
这个文件的作用是汇总前面各部分代码，对文档进行基于lda的相似度计算

'''

from lda_model import get_lda_model
from similarity import lda_similarity_corpus
from save_result import save_similarity_matrix
import traceback

INPUT_PATH = ""
OUTPUT_PATH = "./res/lda_simi_matrix.txt"

def main():
    try:

        # 语料
        documents = ["Shipment of gold damaged in a fire",
                     "Delivery of silver arrived in a silver truck",
                     "Shipment of gold arrived in a truck"]

        # 训练lda模型
        K = 2 # number of topics
        lda_model, _, _,corpus_tf, _ = get_lda_model(documents, K)

        # 计算语聊相似度
        lda_similarity_matrix = lda_similarity_corpus( corpus_tf, lda_model )

        # 保存结果
        save_similarity_matrix( lda_similarity_matrix, OUTPUT_PATH )

    except Exception,e:
        print traceback.print_exc()

main()