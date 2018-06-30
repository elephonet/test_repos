import numpy as np
import pandas as pd
import re
import jieba
import jieba.posseg as pseg
# import codecs
import traceback
from gensim import corpora, models, similarities
pd.set_option('display.width',20000)


def read_raw_df(filename):
    df = pd.read_csv(filename, encoding='gb18030')
    #df['id'] = df.index
    df['经营范围'] = df['经营范围'].map(lambda x: re.sub('\（(.*?)\）', '', str(x)))
    return df

def add_word(filename):
    for word in open(filename,'r').readlines():
        jieba.add_word(word.strip())

#add_word('my_own_dic.txt')

def add_words(filename):
    words = []
    for word in open(filename,'r').readlines():
        words.append(word.strip())
    return list(set(words))

def read_key_words(filename):
    lis = ['1:新一代信息技术','2:高端装备','3:新材料','4:生物产业','5:新能源汽车','6:新能源产业','7:节能环保','8:数字创意']
    dic = {}
    i = 0
    for topic in open(filename).read().split(','):
        topic = topic.strip().split('\n')
        dic['%s'%lis[i]] = topic
        i+=1
    #print (dic)
    return dic

#read_key_words('my_own_dic.txt')

def corpus_label(filename):
    corpus = []
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v','m']
    for line in open(filename,'r').readlines():
        if line is not '\n':
            line_word = [word for word,type in pseg.cut(line) if type not in stop_flag]
            #line_word = jieba.cut(line)
            corpus.extend(list(set(line_word)))
        #print (corpus)
    return list(set(corpus))



def corpus_enterprice(filename):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r', 'v', 'm']
    df = pd.read_csv(filename, encoding='gb18030')
    df['id'] = df.index
    df['经营范围'] = df['经营范围'].map(lambda x: re.sub('\（(.*?)\）', '', str(x)))
    #print(df['经营范围'].head(10))
    df['经营 范围'] = df['经营范围'].map(lambda x: [k for k,v in pseg.cut(x) if v not in stop_flag])
    df['经营 范围'] = df['经营 范围'].map(lambda x: list(set(x)))
    return df

def f(x,inter_sec_dic):
    dic = {}
    for k,v in inter_sec_dic.items():
        inter = list(set(x).intersection(set(v)))
        if inter != []:
            dic[k] = inter

    return dic


def find_key_word(df,inter_sec_dic):
    df['inter_sec'] = df['经营 范围'].map(lambda x: f(x,inter_sec_dic))
    df['topic_num'] = df.inter_sec.map(lambda x: len(x))
    #df['inter_sec'] = df.inter_sec.fillna(-999)
    return df


def main():
    add_word('my_own_dic.txt')
    corpus = read_key_words('my_own_dic.txt')
    #print (corpus)
    df0 = read_raw_df('base_add_data.csv')

    df = corpus_enterprice('base_add_data.csv')

    df1 = find_key_word(df,corpus)
    #df = df[df.inter_sec!=dict({})]
    #df_topic = df1[df1.topic_num>0]

    #print (df_topic) #[['id','经营范围','inter_sec','topic_num']]

    #df_te = df0[df0.index.isin(df_topic.id)]

    #print (df_te)
    df1.to_csv('base_add_data_matched.csv',index=None,encoding='utf-8')
# list(set(a).intersection(set(b)))
main()










#jieba.load_userdict(file_name)

#corpus_x = corpus_enterprice('base_add_data.csv')
#corpus_y = corpus_label('enterprice_cluster_corpus1.txt')

#print (corpus_x[:9])
#print (corpus_y[:2])

#corpus_y1 = corpus_label1('enterprice_cluster_corpus1.txt')
#print (corpus_y1[:2])