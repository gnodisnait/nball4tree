import numpy as np
import pprint


def simCos(word, dic, num=5, ball=True):
    vLst = []
    for k, v in dic.items():
        if word != k:
            if ball:
                vLst.append([word, k, np.dot(dic[word][:-2], dic[k][:-2])])
            else:
                vLst.append([word, k, np.dot(dic[word], dic[k])])

    vLst = sorted(vLst,key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def nearest_neighbors_of_word_sense(tlst = None, dic=None, simFunc=simCos, numOfNeighbors=0, isBall=True):
    neigbors = {}
    keys = list(dic.keys())
    for word in tlst:
        if word in keys:
            neigbors[word] =[ele[0] for ele in simFunc(word, dic, num=numOfNeighbors, ball=isBall)]

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(neigbors)
    return neigbors