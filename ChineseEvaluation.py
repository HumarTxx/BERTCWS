#! -*- coding: utf-8 -*-

import re

def to_region(segmentation: str) -> list:
     """
     将分词结果转换为区间
     :param segmentation: 商品 和 服务
     :return: [(0, 2), (2, 3), (3, 5)]
     """
     region = []
     start = 0
     for word in re.compile("\\s+").split(segmentation.strip()):
         end = start + len(word)
         region.append((start, end))
         start = end
     return region

def prf(gold: str, pred: str, dic) -> tuple:
     """
     计算P、R、F1
     :param gold: 标准答案文件，比如“商品 和 服务”
     :param pred: 分词结果文件，比如“商品 和服 务”
     :param dic: 词典
     :return: (P, R, F1, OOV_R, IV_R)
     """
     A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
     A, B = set(to_region(gold)), set(to_region(pred))
     A_size += len(A)
     B_size += len(B)
     A_cap_B_size += len(A & B)
     text = re.sub("\\s+", "", gold)

     for (start, end) in A:
         word = text[start: end]
         if word in dic:
             IV += 1
         else:
             OOV += 1

     for (start, end) in A & B:
         word = text[start: end]
         if word in dic:
             IV_R += 1
         else:
             OOV_R += 1
     p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
     return p, r, 2 * p * r / (p + r), OOV_R / OOV * 100, IV_R / IV * 100


if __name__ == '__main__':
     dic = ['结婚', '尚未', '的', '和', '青年', '都', '应该', '好好考虑', '自己',  '人生', '大事']
     gold = '结婚 的 和 尚未 结婚 的 都 应该 好好 考虑 一下 人生 大事'
     pred = '结婚 的 和尚 未结婚 的 都 应该 好好考虑 一下 人生大事'
     print("Precision:%.2f ------ Recall:%.2f------- F1:%.2f ------ OOV-R:%.2f-------- IV-R:%.2f" % prf(gold, pred, dic))
