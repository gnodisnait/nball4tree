import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import decimal
import itertools
import os
from nltk.corpus import wordnet as wn
from collections import defaultdict
from pprint import pprint
from nball4tree.util_vec import average_vector, dis_between_norm_vec, spearmanr
from nball4tree.config import DECIMAL_PRECISION

decimal.getcontext().prec = DECIMAL_PRECISION

UpperCats = defaultdict(list)


def maximum_deviation(ballStemFile="", word2vecFile="", ballFile="", dim=50, ofile=""):
    ballStemsDic = defaultdict()
    stdlines = []
    data = defaultdict()
    word2vecDic = defaultdict()

    with open(word2vecFile, 'r') as w2v:
        for line in w2v.readlines():
            wlst = line.strip().split()
            word2vecDic[wlst[0]] = vec_norm([decimal.Decimal(ele) for ele in wlst[1:]])

    with open(ballStemFile, 'r') as ifh:
        for ln in ifh:
            w=ln[:-1].strip()
            ballStemsDic[w] = []

    dic = get_word_sense_dic(ballFile)
    for key, value in dic.items():
        w = key.split('.')[0]
        ballStemsDic[w].append(vec_norm(value[:dim]))

    for key, values in ballStemsDic.items():
        preTrained = word2vecDic[key]
        disLst = [dis_between_norm_vec(p, preTrained) for p in values]
        val = decimal.Decimal(np.std(disLst, axis=0))
        val = float(val.quantize(decimal.Decimal('.000000000000000000001')))
        if val in data:
            data[val] += 1
        else:
            data[val] = 1

        stdlines.append(' '.join([key, str(val)]))

    lists = sorted(data.items())
    x, y = zip(*lists)
    plt.scatter(x, y, color='red')
    plt.title('consistency analysis')
    plt.xlabel('standard deviation of word-stems')
    plt.ylabel('count')
    plt.show()

    keys = list(data.keys())
    dicStd = {
        "max Std":   max(keys),
        "Std > 0.2": sum([data[e] for e in [k for k in keys if k>0.2]]),
        "Std <= 0.2 and >0.1": sum([data[e] for e in [k for k in keys if k> 0.1 and k<= 0.2]]),
        "Std <=0.1 and > 10^(-12)": sum([data[e] for e in [k for k in keys if k > 10 ** -12 and k <= 0.1]]),
        "Std <=10^(-12) and >0": sum([data[e] for e in [k for k in keys if k > 0 and k <= 10 ** -12]]),
        "Std == 0": sum([data[e] for e in [k for k in keys if k == 0]])
    }
    pprint(dicStd)

    with open(ofile, 'w') as ofh:
        ofh.write('\n'.join(stdlines))


def test_with_wordsim353(testFile, ballDic=None, simWordFunc=None, wsSimFunc=None):
    measureLstByMachine = []
    measureLstByHuman = []
    dic = get_word_sense_dic(ballDic)

    with open(testFile, 'r') as wsfh:
        for line in wsfh.readlines():
            wlst = line.split() # word ~ word
            wslst1 = get_all_ws(wlst[0], WSDic=dic)
            wslst2 = get_all_ws(wlst[1], WSDic=dic)
            if wslst1 and wslst2:
                print('*ws lists:', wslst1, wslst2)
                machineValue = simWordFunc(wslst1, wslst2, wsSimFunc=wsSimFunc, WSDic=dic)
                measureLstByHuman.append(decimal.Decimal(wlst[2]))
                measureLstByMachine.append(machineValue)

    evls = spearmanr(measureLstByMachine, measureLstByHuman)
    print(evls)
    return evls


def test_wordembedding_part_using_wordsim353(testFile, ballDic=None, weDic=None, dim=50, outfile=None):
    measureLstByMachine = []
    measureLstByMachine1 = []
    measureLstByHuman = []
    olines = []
    dic = get_word_sense_dic(ballDic)
    dic1 = get_word_sense_dic(weDic)

    with open(testFile, 'r') as wsfh:
        N = 0
        negLst  = []
        for line in wsfh.readlines():
            wlst = line.split() # word ~ word

            ws1 = get_all_ws(wlst[0], WSDic=dic)
            if not ws1:
                print('*', wlst[0])
                negLst.append(wlst[0])
                continue
            vecLst1 = [dic[w][:dim] for w in ws1]
            wordembedding1 = average_vector(vecLst1)

            ws2 = get_all_ws(wlst[1], WSDic=dic)


            if not ws2:
                print('**', wlst[1])
                negLst.append(wlst[1])
                continue
            vecLst2 = [dic[w][:dim] for w in ws2]
            wordembedding2 = average_vector(vecLst2)

            csim = np.dot(wordembedding1, wordembedding2)
            N += 1
            measureLstByMachine.append(csim)
            measureLstByMachine1.append(np.dot(dic1[ws1[0].split('.')[0]],dic1[ws2[0].split('.')[0]]))
            measureLstByHuman.append(decimal.Decimal(wlst[2]))
            olines.append(line)

    evls = spearmanr(measureLstByMachine, measureLstByHuman)
    print("WE in ball",evls)
    evls = spearmanr(measureLstByMachine1, measureLstByHuman)
    print("WE in ", weDic, evls)
    print("N:", N)
    negLst = set(negLst)
    print('Neglected words', negLst)
    print('Num of neglected words', len(negLst))
    with open(outfile, 'w') as ofh:
        ofh.writelines(olines)
    return evls


def test_wordembedding_part_using_SCWS(testFile, ballDic=None, weDic=None, dim=50, outfile=None):
    measureLstByMachine = []
    measureLstByMachine1 = []
    measureLstByHuman = defaultdict(list)
    olines = []
    dic = get_word_sense_dic(ballDic)
    dic1 = get_word_sense_dic(weDic)

    with open(testFile, 'r') as wsfh:
        N = 0
        negLst  = []
        for line in wsfh.readlines():
            wlst = line.split() # word ~ word

            ws1 = get_all_ws(wlst[1], WSDic=dic)
            if not ws1:
                print('*', wlst[1])
                negLst.append(wlst[1])
                continue
            vecLst1 = [dic[w][:dim] for w in ws1]
            wordembedding1 = average_vector(vecLst1)

            ws2 = get_all_ws(wlst[3], WSDic=dic)
            if not ws2:
                print('**', wlst[3])
                negLst.append(wlst[3])
                continue
            vecLst2 = [dic[w][:dim] for w in ws2]
            wordembedding2 = average_vector(vecLst2)

            csim = np.dot(wordembedding1, wordembedding2)
            N += 1
            measureLstByMachine.append(csim)
            measureLstByMachine1.append(np.dot(dic1[ws1[0].split('.')[0]],dic1[ws2[0].split('.')[0]]))
            humanEvlLst = [decimal.Decimal(ele) for ele in wlst[-10:]]
            for i in range(10):
                measureLstByHuman[i].append(humanEvlLst[i])
            olines.append(line)

    print("N:", N)
    negLst = set(negLst)
    print('Neglected words', negLst)
    with open(outfile, 'w') as ofh:
        ofh.writelines(olines)
    for i in range(10):
        evls = spearmanr(measureLstByMachine, measureLstByHuman[i])
        print("WE in ball",evls)
        evls = spearmanr(measureLstByMachine1, measureLstByHuman[i])
        print("Glove WE", evls)
    return evls


def enlarge_observation_anlge(ele, i):
    if ele == 1:
        return 1
    v = ele * (10**i) - decimal.Decimal(10**i -1)
    if v > 0:
        return v
    else:
        return 0


def maxWSense(wsLst1,wsLst2, WSDic=None, wsSimFunc=None, cat=False):
    """
    :param wd1:
    :param wd2:
    :param WSDic:
    :return:
    """
    sim, upsim = 0, 0
    for ws1, ws2 in itertools.product(wsLst1, wsLst2):
        cSim = wsSimFunc(ws1, ws2,  WSDic=WSDic)
        if cat:
            upperWS1 = get_upper_category(ws1, WSDic=WSDic)
            upperWS2 = get_upper_category(ws2, WSDic=WSDic)
            upsim = wsSimFunc(upperWS1, upperWS2,  WSDic=WSDic)
            cSim = (cSim + upsim)/2
        if cSim > sim:
            sim = cSim
    return sim


def count_frequency(ws):
    """
    count frequency of a word sense
    :param ws:
    :return:
    """
    synset = wn.synset(ws)
    freq = 0
    for lemma in synset.lemmas():
        if lemma.count() == 0:
            freq +=1
        else:
            freq += lemma.count()
    return freq


def sim_qsr2_bb(ball=None, cv=None):
    """
    change! first TOPO, then degree!
    IN: > 10, can be ball in cv, or cv in ball!
    PO: [0,10]
    NO: < 0
    :param ball:
    :param cv:
    :return:
    """
    if len(cv) == 0:
        return 0
    scos = np.dot(ball[:-2], cv[:-2])
    r1,r2 = ball[-1], cv[-1]
    l1,l2 = ball[-2], cv[-2]
    dis = np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*scos)
    mnr = min(r1, r2)
    mxr = max(r1, r2)
    f1 = r1 + r2 -  dis
    f2 = mxr - dis - mnr
    if f1 < 0:
        return f1, 0
    elif f2 > 0 and r1 > r2:
        return f2, -1
    elif f2 >= 0 and r1 <= r2:
        return f2, 1
    else:
        return f1, f2


def get_upper_category(word, dic):
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr2_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k, simv])
    print(word, vLst)
    upperLst = sorted(filter(lambda ele: ele[2][0] > 0, vLst), key=lambda x: x[2][0])[:1]
    return upperLst[0][0]


def get_upper_cats(word, dic, cat=20):
    global UpperCats
    rlt = UpperCats[word]
    if rlt:
        return rlt
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr2_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k]+list(simv))

    if vLst:
        vLst = filter(lambda ele:ele[3]==1, vLst)
        vLst = sorted(vLst, key=lambda x:x[2])
        UpperCats[word] = [ele[1] for ele in vLst[:cat]]
        return [ele[1] for ele in vLst[:cat]]
    else:
        return []


def weightWSense(wsLst1, wsLst2, WSDic=None, wsSimFunc=None, cat=False):
    """
    :param wd1:
    :param wd2:
    :param WSDic:
    :return:
    """
    wsFreqDic = {ws:count_frequency(ws) for ws in wsLst1 + wsLst2}
    wsSum1 = sum(wsFreqDic[ws] for ws in wsLst1)
    wsSum2 = sum(wsFreqDic[ws] for ws in wsLst2)
    sim = 0
    for ws1, ws2 in itertools.product(wsLst1, wsLst2):
        cSim = wsSimFunc(ws1,ws2, WSDic=WSDic)
        if cat:
            upperWS1 = get_upper_category(ws1, WSDic=WSDic)
            upperWS2 = get_upper_category(ws2, WSDic=WSDic)
            upsim = wsSimFunc(upperWS1,upperWS2, WSDic=WSDic)
            sim += wsFreqDic[ws1] * wsFreqDic[ws2] * (cSim+upsim)/2/ wsSum1 / wsSum2
        else:
            sim += wsFreqDic[ws1] * wsFreqDic[ws2] * cSim / wsSum1 / wsSum2
    return sim


def get_all_ws(word, WSDic=None):
    return [ele for ele in WSDic.keys() if ele.startswith(word.lower()+".")]


def get_word_sense_dic(wordEmbeddingFile):
    dic=defaultdict()
    with open (wordEmbeddingFile, 'r') as fh:
        for line in fh.readlines():
            lst = line.split()
            v = [decimal.Decimal(ele) for ele in lst[1:]]
            dic[lst[0]] = v
    return dic


def vec_norm(v):
    return list(np.divide(v, np.linalg.norm(v)))


def sim_cosine(ws1, ws2, WSDic = None):
    ball1 = WSDic[ws1]
    ball2 = WSDic[ws2]
    return np.dot(vec_norm(ball1[:-2][:50]), vec_norm(ball2[:-2][:50]))


def ratio2(a,b):
    return np.sqrt(2*a*b/(a*a+b*b))


def sim_cosine_r(ws1, ws2, WSDic = None):
    ball1 = WSDic[ws1]
    ball2 = WSDic[ws2]
    r1,r2 = ball1[-1], ball2[-1]
    return sim_cosine(ws1,ws2,WSDic=WSDic) * ratio2(r1, r2)


def sim_cosine_lr(ws1, ws2, WSDic = None):
    ball1 = WSDic[ws1]
    ball2 = WSDic[ws2]
    l1, r1 = ball1[-2:]
    l2, r2 = ball2[-2:]
    return sim_cosine(ws1,ws2,WSDic=WSDic) * ratio2(r1, r2)*ratio2(l1,l2)


def dis_between_centres(ball1=None, ball2=None):
    scos = np.dot(ball1[:-2], ball2[:-2])
    l1,l2 = ball1[-2], ball2[-2]
    delta = l1*l1 + l2*l2 - 2*l1*l2*scos
    if delta < 0:
        return 0
    else:
        return np.sqrt(delta)


def sim_qsr(ws1, ws2, WSDic = None):
    ball1 = WSDic[ws1]
    ball2 = WSDic[ws2]
    l1, r1 = ball1[-2:]
    l2, r2 = ball2[-2:]
    dis = dis_between_centres(ball1=ball1, ball2=ball2)
    f = r1+r2-dis
    return f


def sim_qsr2_bb(ball=None, cv=None):
    """
    change! first TOPO, then degree!
    IN: > 10, can be ball in cv, or cv in ball!
    PO: [0,10]
    NO: < 0
    :param ball:
    :param cv:
    :return:
    """
    if len(cv) == 0:
        return 0
    scos = np.dot(ball[:-2], cv[:-2])
    r1,r2 = ball[-1], cv[-1]
    l1,l2 = ball[-2], cv[-2]
    dis = np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*scos)
    mnr = min(r1, r2)
    mxr = max(r1, r2)
    f1 = r1 + r2 - dis
    f2 = mxr - dis - mnr
    if f1 < 0:
        return f1, 0
    elif f2 > 0 and r1 > r2:
        return f2, -1
    elif f2 > 0 and r1 <= r2:
        return f2, 1
    else:
        return f1, f2


def sim_qsr1_bb(ball=None, cv=None):
    """
    change! first TOPO, then degree!
    IN: > 10, can be ball in cv, or cv in ball!
    PO: [0,10]
    NO: < 0
    :param ball:
    :param cv:
    :return:
    """
    if len(cv) == 0:
        return 0
    scos = np.dot(ball[:-2], cv[:-2])
    r1,r2 = ball[-1], cv[-1]
    l1,l2 = ball[-2], cv[-2]
    dis = np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*scos)
    f1 = r1 + r2 - dis
    return f1


def num_connected_cat(ws1, ws2, WSDic = None):
    lst = get_upper_cats(ws1, WSDic)
    cat1 = set(lst)
    lst = get_upper_cats(ws2, WSDic)
    cat2 = set(lst)
    iset = cat1.intersection(cat2)
    setd1 = cat1.difference(cat2)
    setd2 = cat2.difference(cat1)
    if len(iset) == 0:
        return 1
    else:
        num = len(setd1) + len(setd2) + 1
        return 30 - num
