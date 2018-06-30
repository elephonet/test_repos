import numpy as np
import pandas as pd
import re
import jieba
import jieba.posseg as pseg
pd.set_option('display.width',20000)


df = pd.read_csv('base_add_data.csv',encoding='gb18030')
df['经营范围'] = df['经营范围'].map(lambda x: re.sub('\（(.*?)\）','',str(x)))
df1=df.sample(n=100)
#print (df1['经营范围'])


test = '网络信息安全产品和服务48	网络与信息安全硬件	网络与信息安全硬件销售49	网络与信息安全软件	网络与信息安全软件开发50	网络与信息安全服务	网络与信息安全服务'
line_word = [i for i in jieba.cut(test)]

print(line_word)

def add_word(filename):
    for word in open(filename,'r').readlines():
        jieba.add_word(word.strip())
add_word('my_own_dic.txt')

#jieba.load_userdict('my_own_dic.txt')
jieba.add_word('网络与信息')

line_word1 = [i for i in pseg.cut(test)]

print(line_word1)