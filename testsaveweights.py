# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import json
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
import glob 
from tqdm import tqdm
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

chars=open('char.txt').read().decode('utf-8')
model =model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

a=model.get_config()
print a 

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
 


#print 'char==',chars


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


f = open('/home/cugdeeplearn/cug/ChineseWS/result412.txt','w')

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
