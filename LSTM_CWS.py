# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm 
import os

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   

#加上了一个x标签，用来表示不够32字的部分，比如句子是20字的，那么第21～32个标签均为x。

# 
s = open('/home/cugdeeplearn/cug/ChineseWS/corpus/msr_train_utf8.txt').read().decode('utf-8')
#s = open('msr_train.txt').read().decode('utf-8')
s = s.split('\r\n')
 
def clean(s): #整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s
 
print('==start==')

s = u''.join(map(clean, s)) # function clean work for s(every char)
s = re.split(u'[，。！？、]/[bems]', s)
 
data = [] #生成训练样本
label = [] # product label
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])
 
f_data = open('/home/cugdeeplearn/cug/ChineseWS/data.txt','w')
f_label = open('/home/cugdeeplearn/cug/ChineseWS/lable.txt','w')
f_char= open('/home/cugdeeplearn/cug/ChineseWS/char.txt','w')
f_d= open('/home/cugdeeplearn/cug/ChineseWS/d.txt','w')



for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        #f_data.write(x[0])
        label.append(x[1])
        #f_label.write(x[1])

f_data.write(str(data))
f_label.write(str(label))


 #设计模型
#word_size = 128
word_size=256
#maxlen = 32
maxlen=32


d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
print data,label
print d
#[u'\u4eba', u'\u4eec', u'\u5e38', u'\u8bf4',]
#print data
#[u'b', u'e', u's', u's', u'b', u'e', u's', u's', u's', u'b', u'm', u'e'], 
#print label

d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})
 
 
chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)
 


chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)
 
#生成适合模型输入的格式
from keras.utils import np_utils
# [101, 74, 57, 41, 12, 103, 8, 2, 60, 16, 4, 20...

#lambda function
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
f_char.write(str(chars))
print '====!!!!!==='

#print tag[x]#.reshape((-1,1))


#print map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))

#[[[0.0, 1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0,... 
#try:
d['y'] = d['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))
#except Exception as e:
    #print Exception,':',e


#f.write(str(d))
 

from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
 
sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
#blstm = Bidirectional(LSTM(64))
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 

print d
batch_size = 1024
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
 
import h5py 
from keras.models import model_from_json
json_string = model.to_json()  
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')


#读取model  
model = model_from_json(open('my_model_architecture.json').read())  
model.load_weights('my_model_weights.h5')

print('读取model  ')
print model.summary()

#转移概率，单纯用了等概率
zy = {'be':0.5,
      'bm':0.5,
      'eb':0.5,
      'es':0.5,
      'me':0.5,
      'mm':0.5,
      'sb':0.5,
      'ss':0.5
     }
 
zy = {i:np.log(zy[i]) for i in zy.keys()}


 
def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]
 


print 'char==',chars


def simple_cut(s):
    if s:
        print s #liyong
        #print 'char==',chars


#按batch获得输入数据对应的输出，函数的返回值是预测值的numpy array
#predict(self, x, batch_size=32, verbose=0)
        #r = model.predict(np.array([list(chars[list(s)])+[0]*(maxlen-len(s))]),  verbose=False)[0][:len(s)]
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        #r = model.predict(np.array([list(chars[list(s)])+[0]*(maxlen-len(s))]), verbose=False)
        r = np.log(r)

        #print "r",r

        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []
 
not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')


def cut_words(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):

        #print 'question===',s[j:i.start()]#liyong

        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


print 'start---test'


f = open('/home/cugdeeplearn/cug/ChineseWS/result414.txt','w')

print '-start-'

result=[]
result_txt=[]
temp=[]


txt_names = glob.glob('./newTest/*.txt')
 

#stops = u'，。！？；、：,\.!\?;:\n'
stops = u'\n'

for name in tqdm(iter(txt_names)):
    txt = open(name).read().decode('utf-8', 'ignore')
    txt = re.sub('/[a-z\d]*|\[|\]', '', txt)
    txt = [i.strip(' ') for i in re.split('['+stops+']', txt) if i.strip(' ')]
    for line in txt:
        line = line.decode('utf-8', 'ignore')
        try:
            temp = ' /'.join(cut_words(line))
            f.write(' '.join(temp))

        except Exception as e:
            print Exception,':',e


#print   "".join(list(result))
    #print result