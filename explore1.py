import numpy as np
import pandas as pd
import re
import jieba
import jieba.posseg as pseg
# import codecs
import traceback
from gensim import corpora, models, similarities
pd.set_option('display.width',20000)

def corpus_label(filename):
    corpus = []
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v','m']
    for line in open(filename,'r').readlines():
        if line is not '\n':
            line_word = [word for word,type in pseg.cut(line) if type not in stop_flag]
            corpus.append(line_word)
        #print (corpus)
    return corpus


#corpus_label('enterprice_cluster_corpus1.txt')


def pull(df):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v', 'm']
    #df_new = pd.DataFrame({'id': [df.ix[0,'id']]  * df.ix[0,'work_range_len'], 'work_range': df.ix[0,'经营范围'].split('；')})
    df_new = pd.DataFrame({'id':[],'work_range':[]})
    for i in range(len(df)):
        df_add = pd.DataFrame({'id': [df.ix[i,'id']]  * df.ix[i,'work_range_len'],
                               'work_range': df.ix[i,'经营范围'].split('；')})
        df_add['work_range'] = df_add.work_range.map(lambda x: [v for v,k in pseg.cut(x) if k not in stop_flag])
        df_new = pd.concat([df_new,df_add])
        corpus = df_new.work_range.tolist()
    #print (df_new.work_range.tolist()[:5])
    return corpus


def corpus_enterprice(filename):

    df = pd.read_csv(filename, encoding='gb18030',nrows=1000)
    df['id'] = df.index
    df['经营范围'] = df['经营范围'].map(lambda x: re.sub('\（(.*?)\）', '', str(x)))
    #print(df['经营范围'].head(10))
    df['work_range_len'] = df['经营范围'].map(lambda x: len(x.split('；')))
    #print (df['work_range_len'].head(10))
    corpus = pull(df)
    #print (df_new.head(30))
    return corpus


#corpus_enterprice('base_add_data.csv')
def corpus_combine(filename_x,filename_y):
    corpus = []
    corpus_x = corpus_enterprice(filename_x)
    corpus_y = corpus_label(filename_y)
    corpus.extend(corpus_y)
    corpus.extend(corpus_x)
    return corpus



# 训练tf_idf模型
def tf_idf_trainning(corpus):
    try:

        # 将所有文章的token_list映射为 vsm空间
        dictionary = corpora.Dictionary(corpus)

        # 每篇document在vsm上的tf表示
        corpus_tf = [ dictionary.doc2bow(token_list) for token_list in corpus ]

        # 用corpus_tf作为特征，训练tf_idf_model
        tf_idf_model = models.TfidfModel(corpus_tf)

        # 每篇document在vsm上的tf-idf表示
        corpus_tfidf = tf_idf_model[corpus_tf]

        #print "[INFO]: tf_idf_trainning is finished!"
        return dictionary, corpus_tf, corpus_tfidf

    except Exception:
        print (traceback.print_exc())

# 训练lda模型
def lda_trainning( dictionary, corpus_tfidf, K ):
    try:

        # 用corpus_tfidf作为特征，训练lda_model
        lda_model = models.LdaModel( corpus_tfidf, id2word=dictionary, num_topics = K )

        # 每篇document在K维空间上表示
        corpus_lda = lda_model[corpus_tfidf]

        #print "[INFO]: lda_trainning is finished!"
        return lda_model, corpus_lda

    except Exception:
        print (traceback.print_exc())


def get_lda_model(filename_x, filename_y,K ):
    try:

        corpus = corpus_combine(filename_x,filename_y)
        # 文档预处理
        #documents_token_list = documents_pre_process( documents )

        # 获取文档的字典vsm空间,文档vsm_tf表示,文档vsm_tfidf表示
        dict, corpus_tf, corpus_tfidf = tf_idf_trainning(corpus)

        # 获取lda模型,以及文档vsm_lda表示
        lda_model, corpus_lda = lda_trainning( dict, corpus_tfidf, K )

        #print "[INFO]:get_lda_model is finished!"
        return lda_model, corpus_lda, dict, corpus_tf, corpus_tfidf

    except Exception:
        print (traceback.print_exc())

#lda_model, corpus_lda, dict, corpus_tf, corpus_tfidf = get_lda_model('base_add_data.csv','enterprice_cluster_corpus1.txt' ,10 )
#print (corpus_lda[2])
#print (dict)
#print (corpus_tfidf[6])




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

    except Exception:
        print (traceback.print_exc())




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

        #print "[INFO]:lda_similarity_corpus is finished!"
        return lda_similarity_matrix

    except Exception:
        print (traceback.print_exc())


def main():
    try:
        lda_model, corpus_lda, dictionary, corpus_tf, corpus_tfidf = get_lda_model('base_add_data.csv',
                                                                             'enterprice_cluster_corpus1.txt', 10)


        raw_corpus = corpus_combine('base_add_data.csv','enterprice_cluster_corpus1.txt')

        index_bow = [dictionary.doc2bow(label) for label in raw_corpus[:9]]
        index = similarities.MatrixSimilarity(lda_model[index_bow])
        for i in range(len(raw_corpus[9:])):
        #for word in [['人工智能'],['节能环保'],['信息安全'],['新能源汽车']]:
            query_simi_list = []
            query_bow = dictionary.doc2bow(raw_corpus[9+i])
            #query_bow = dictionary.doc2bow(word)
            query_lda = lda_model[query_bow]
            simi = index[query_lda]
            query_simi_list = [item for _, item in enumerate(simi)]
            query_simi_list.append(enumerate(simi))
            print (raw_corpus[9 + i])
            print (word)
            print (query_simi_list)
            print ('\n')

        #lda_similarity_matrix.append(query_simi_list)

    except Exception:
        print (traceback.print_exc())

#main()

raw_corpus = corpus_combine('base_add_data.csv','enterprice_cluster_corpus1.txt')
for i in raw_corpus:
    print (i)