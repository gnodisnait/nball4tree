
# python nearest_neighbors.py /Users/tdong/data/glove/glove.6B/glove.6B.50Xball.V03.centre.txt 0

import os
import numpy as np
import pprint
import decimal
from nball4tree.util_file import get_word_embedding_dic
from nball4tree.config import DECIMAL_PRECISION

decimal.getcontext().prec = DECIMAL_PRECISION


def create_ball_file(ipath=None, oFile=None):
    """
    call: create_ball_file(ipath=wsBallPath, oFile=ballCentreFile)
    :param ipath:
    :param oFile:
    :return:
    """
    open(oFile, 'w').close()
    with open(oFile, 'a') as ofh:
        for fn in [ele for ele in os.listdir(ipath) if len(ele.split('.')) > 2]:
            with open(os.path.join(ipath, fn), 'r') as ifh:
                for ln in ifh:
                    wlst = ln[:-1].split()
                    ofh.write(" ".join([fn]+wlst)+"\n")


def compute_cosine_similarity(pdic, w2vFile=""):
    w2vDic = get_word_embedding_dic(w2vFile)
    rltDic = {}
    for key, lst in pdic.items():
        for ele in lst:
            vec1 = vec_norm(w2vDic[key])
            vec2 = vec_norm(w2vDic[ele])
            rltDic['-'.join([key, ele])] = str(np.dot(vec1, vec2))
    pprint.pprint(rltDic)


def find_minus_cos(word1, file4word2, wordEmbeddingFile, outfile):
    dic = get_word_embedding_dic(wordEmbeddingFile)
    rlt = []
    word1 = word1.lower()
    vec1 = dic.get(word1, [])
    with open(file4word2, 'r') as ifh:
        lst2 = ifh.readlines()
    print(lst2[0], lst2[-1])
    print("+", len(lst2))
    for word2 in lst2:
            word2 = '_'.join(word2.lower().split())
            vec2 = dic.get(word2, [])
            if vec1 and vec2:
                vec1 = vec_norm(vec1)
                vec2 = vec_norm(vec2)
                cos = np.dot(vec1, vec2)
                if cos < 0:
                    rlt.append(word2+" "+str(cos))
    with open(outfile, 'w') as ofh:
        ofh.write('\n'.join(rlt)+'\n')
    print(len(rlt))
    return rlt


def simCos(word, dic, cat=5, num=5, ball=True):
    vLst = []
    for k, v in dic.items():
        if k == len(dic):
            continue
        if word != k:
            if ball:
                vLst.append([word, k, np.dot(dic[word][:-2], dic[k][:-2])])
            else:
                vLst.append([word, k, np.dot(dic[word], dic[k])])

    vLst = sorted(vLst,key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def ratio2(a,b):
    return np.sqrt(2*a*b/(a*a+b*b))


def simCosRR(word, dic, num=5):
    vLst = []
    for k, v in dic.items():
        if word != k:
            vLst.append([word, k, np.dot(dic[word][:-2], dic[k][:-2])*ratio2(dic[word][-1], dic[k][-1])])

    vLst = sorted(vLst, key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def simCosRRLL(word, dic, num=5):
    vLst = []
    for k, v in dic.items():
        if word != k:
            if word == 'berlin.n.01' and k == 'geneva.n.01':
                print('break')
            simv = np.dot(dic[word][:-2], dic[k][:-2])*ratio2(dic[word][-1], dic[k][-1])*ratio2(dic[word][-2], dic[k][-2])
            vLst.append([word, k, simv])

    vLst = sorted(vLst, key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def dis_between_centres(ball=None, cv=None):
    if len(cv) == 0:
        return 0
    scos = np.dot(ball[:-2], cv[:-2])
    l1,l2 = ball[-2], cv[-2]
    dis = np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*scos)
    return dis


def sim_qsr_bb(ball=None, cv=None):
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
    if np.abs(r1-r2) >= dis: # IN (> dis - np.abs(r1-r2))
        return np.abs(r1-r2)- dis
    elif dis > r1 + r2:
        return r1 + r2 - dis
    else: # PO (> 0)
        return dis - np.abs(r1-r2)


def simQSR(word, dic, num=5):
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k, simv])

    vLst = sorted(vLst, key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


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
        return f1, 'dis'
    elif f2 > 0 and r1 > r2:
        return f2, 'contain'
    elif f2 > 0 and r1 <= r2:
        return f2, 'containedBy'


def simQSR2(word, dic, num=5, cat=0):
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr2_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k, simv])

    vLst = sorted(vLst, key=lambda x: x[2][0], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def UppersSimMembers(word, dic, cat=5, num=0):
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr2_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k, simv])

    # vLst = list(filter(lambda ele: ele[2]!=None, vLst))

    upperLst = sorted(filter(lambda ele: ele[2][0]>0, vLst), key=lambda x: x[2][0])[:cat]
    lst = []
    for i in range(len(upperLst)):
        lst += [upperLst[i][1]+'$+_'+str(i+1)+'$']
    catStr = ' '.join(lst)
    memberLst = sorted(filter(lambda ele: ele[2][0]<0, vLst), key=lambda x: x[2][0], reverse=True)[:num]
    closestMem = memberLst[0][2][0]
    lst = []
    for i in range(len(memberLst)):
        ele = memberLst[i]
        vLst0 = []
        for k0, v0 in dic.items():
            if memberLst[i][1] != k0:
                simv = sim_qsr2_bb(ball=dic[memberLst[i][1]], cv=dic[k0])
                vLst0.append([memberLst[i][1], k0, simv])
        # vLst0 = list(filter(lambda ele: ele[2] != None, vLst0))
        mlst = sorted(filter(lambda ele: ele[2][0] < 0, vLst0), key=lambda x: x[2][0], reverse=True)[0]
        closestMem1 = mlst[2][0]
        lst += [memberLst[i][1]+' ($-_'+str(i+1)+'$'+ ' {0:.2f}),'.format(max(closestMem, closestMem1)/ele[2][0])]
    sibStr = ' '.join(lst)
    #return [[ele[1], ele[2]] for ele in upperLst + memberLst]
    return {'cat': catStr, 'sib': sibStr}


def get_upper_cats(word, dic, cat=20):
    vLst = []
    for k, v in dic.items():
        if word != k:
            simv = sim_qsr2_bb(ball=dic[word], cv=dic[k])
            vLst.append([word, k, simv])

    upperLst = sorted(filter(lambda ele: ele[2][0] > 0, vLst), key=lambda x: x[2][0])[:cat] 
    return [[ele[1]] for ele in upperLst]


def simDis(word, dic, num=5, cat=0):
    vLst = []
    for k, v in dic.items():
        if word != k:
            dis = dis_between_centres(ball=dic[word], cv=dic[k])
            vLst.append([word, k, dis])

    vLst = sorted(vLst, key=lambda x: x[2], reverse=True)
    rlt = vLst[:min(num, len(vLst))]
    return [[ele[1], ele[2]] for ele in rlt]


def get_polysemy(w, keys):
    """
    w is word, 'bat'
    keys is a list of word with part of speech: 'bat_nn', 'bat_vbz'
    :param w:
    :param keys:
    :return: a list of all words w with all POS
    """
    return [ele for ele in keys if ele.startswith(w+"_")]


def get_words_from_test_file(nFile):
    words = []
    with open(nFile, 'r') as wsfh:
        for line in wsfh.readlines():
            for w in line.split()[:1]:
                if w not in words:
                    words.append(w)
    return words


def test_with_neighbors_of_word_sense(tlst = None, dicFile = None, testFile=None, simFunc=None, cat=5,
                                      numOfNeighbors=0, isBall=True):
    dic = get_word_embedding_dic(dicFile)
    neigbors = {}
    keys = list(dic.keys())
    if tlst is None:
        tlst = get_words_from_test_file(testFile)
    for word in tlst:
        if word in keys:
            neigbors[word] = simFunc(word, dic, cat=cat, num=numOfNeighbors, ball=isBall)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(neigbors)
    return neigbors


def qsr_IN_degree(ball=None, cv=None):
    """
    :param ball:
    :param cv:
    :return:
    """
    scos = np.dot(ball[:-2], cv[:-2])
    r1,r2 = ball[-1], cv[-1]
    l1,l2 = ball[-2], cv[-2]
    dis = np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*scos)
    return r2 - dis - r1


def vec_norm(v):
    return list(np.divide(v, np.linalg.norm(v)))


def construct_leaf_ball(ws, wd2vecDic=None, cat=None):
    global L0, R0, addDim
    word = ws.split('.')[0]
    w2v = wd2vecDic(word)
    cpoint = vec_norm(w2v) + [ele + 1 for ele in cat]
    return vec_norm(cpoint) + [L0, R0]


if __name__ == "__main__":
    word2vecFile = "/Users/tdong/data/glove/glove.6B/glove.6B.50d.txt"
    conceptNet2vecFile = "/Users/tdong/data/conceptnet/numberbatch-en-17.06.txt"
    ballFile = "/Users/tdong/data/glove/glove.6B/glove.6B.50Xball.V10.txt"
    wsfpath = "/Users/tdong/data/glove/glove.6B"
    wsBallPath = os.path.join(wsfpath, "glove.6B.50Xball.V10")

    # create_ball_file(ipath=wsBallPath, oFile=ballCentreFile)
    # tlst = ["france.n.01", "france.n.02","bank.n.01", "bank.n.03", "bank.v.01", "bank.n.04", "bank.n.05", "bank.n.06",
    #        "love.n.01", "love.n.02", "love.v.01", "love.v.02", "love.v.03",
    #        "stock.v.01","stock.v.02", "stock.v.03", "stock.v.04", "stock.n.01", "stock.n.02",
    #        "stock.n.03", "star.n.01", "star.n.03", "star.n.04", "star.n.05", "star.v.01", "star.v.02", "star.v.03",
    #        "plant.n.01", "plant.n.02", "plant.n.03", "plant.n.04", "plant.v.01", "plant.v.04",
    #        "plant.v.05", "plant.v.06"]
    # tlst = ['riyadh.n.01','city.n.01']
    tlst =["france.n.01", "france.n.02",'beijing.n.01', 'berlin.n.01','berlin.n.02',  'y.n.02',
           'tiger.n.01', 'cat.n.01']
    tlst = ['philosopher.n.01']
    tlst = ['beijing', 'berlin']
    # tlst = ["bank.n.01", "bank.n.03", "bank.v.01", "bank.n.04", "bank.n.05", "bank.n.06",
    #        'tiger.n.01', 'cat.n.01', 'book.n.06', 'paper.n.01']
    # tlst = ["france.n.01", "france.n.02","england.n.01",
    #        "china.n.01",  "china.n.02", "rome.n.01","madrid.n.01", "prague.n.01", "paris.n.01", "paris.n.02", "paris.n.03",
    #        "paris.n.04", "thessaloniki.n.01"]
    # tlst = ['simon.n.02', 'berlin.n.02']
    # tlst = ['manchester.n.01','tokyo.n.01', "seattle.n.01", "", 'shanghai.n.01', 'shanghai.v.01']
    # tlst = ['simon.n.02']#, 'williams.n.01', 'foster.n.01','dylan.n.01','mccartney.n.01']
    test_with_neighbors_of_word_sense(tlst=tlst, dicFile=conceptNet2vecFile, simFunc=simCos, cat=8,
                                      numOfNeighbors=10, isBall=False)
    # find_minus_cos("china",
    #               "/Users/tdong/data/glove/glove.6B/China_City.txt",
    #               "/Users/tdong/data/glove/glove.6B/glove.6B.50d.txt",
    #               "/Users/tdong/data/glove/glove.6B/china_city_minus.txt"
    #               )
    pdic = {'beijing': ['london', 'paris', 'washington', 'atlanta', 'potomac', 'boston', 'baghdad'],
           'berlin': ['madrid', 'toronto', 'rome', 'columbia', 'sydney',
                       'dallas', 'simon','williams', 'foster', 'dylan', 'mccartney', 'lennon'],
            #'france': ['russia', 'japan', 'kingdom', 'india', 'blighty', 'israel',
            #           'white', 'poet', 'uhland', 'london', 'journalist', 'woollcott'],
            #'cat': ['tiger', 'fox', 'wolf', 'wildcat', 'tigress', 'vixen'],
            #'y': ['q', 'delta', 'n', 'p', 'f', 'g']
            }
    # pdic = {'tiger': ['survivor', 'neighbor', 'immune', 'linguist', 'bilingual', 'warrior']}
    # compute_cosine_similarity(pdic, w2vFile=conceptNet2vecFile)



