#! -*- coding: utf-8 -*-


import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import uniout

# BERT配置
config_path = 'bert/bert_config.json'
checkpoint_path = 'bert/bert_model.ckpt'
dict_path = 'bert/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 文本编码
text = u'大肠杆菌是人和许多动物肠道中最主要且数量最多的一种细菌'
token_ids, segment_ids = tokenizer.encode(text)
length = len(token_ids) - 2


def dist(x, y):
    """距离函数（默认用欧氏距离）
    可以尝试换用内积或者cos距离，结果差不多。
    """
    return np.sqrt(((x - y)**2).sum())


batch_token_ids = np.array([token_ids] * (2 * length - 1))
batch_segment_ids = np.zeros_like(batch_token_ids)

for i in range(length):
    if i > 0:
        batch_token_ids[2 * i - 1, i] = tokenizer._token_mask_id
        batch_token_ids[2 * i - 1, i + 1] = tokenizer._token_mask_id
    batch_token_ids[2 * i, i + 1] = tokenizer._token_mask_id

vectors = model.predict([batch_token_ids, batch_segment_ids])

threshold = 8
word_token_ids = [[token_ids[1]]]
for i in range(1, length):
    d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
    d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
    d = (d1 + d2) / 2
    if d >= threshold:
        word_token_ids[-1].append(token_ids[i + 1])
    else:
        word_token_ids.append([token_ids[i + 1]])

words = [tokenizer.decode(ids) for ids in word_token_ids]
print(words)
