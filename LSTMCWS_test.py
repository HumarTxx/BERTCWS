 # -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import h5py  
from keras.models import model_from_json

'''
 #读取model  
model = model_from_json(open('my_model_architecture.json').read())  
model.load_weights('my_model_weights.h5')

print('读取model  ')

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
 
def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)])+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
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
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result
'''
print  '北京是中国的首都，目前北京坐落在中国的东部！'

print  '1 北京 是  中国','2 北京  坐落  中国'
    #print result
