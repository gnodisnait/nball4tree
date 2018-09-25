import random
import math
import os


def create_word_sense_list_file(ifile = '', ofile=''):
    rlt = []
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            rlt += ln[:-1].split()
    rlt = list(set(rlt))
    sorted(rlt)
    with open(ofile, 'w') as ofh:
        ofh.write('\n'.join(rlt))


def extract_words_not_in_kg(w2vFile='', iTaxoFile='', oWordsNotinKG=''):
    if not os.path.isfile(w2vFile):
        print('w2v file does not exist:', w2vFile)
        return
    if not os.path.isfile(iTaxoFile):
        print('TaxoFile file does not exist:', iTaxoFile)
        return

    words = []
    with open(w2vFile, 'r') as w2v:
        for line in w2v.readlines():
            wlst = line.split()
            words.append(wlst[0])
    kgEntities = []
    with open(iTaxoFile, 'r') as wsfh:
        for line in wsfh.readlines():
            ws = line.split()[0]
            kgEntities.append(ws.split('.')[0])
    unknownWords = list(set(words).difference(set(kgEntities)))
    if oWordsNotinKG:
        with open(oWordsNotinKG, 'w') as ofh:
            ofh.write('\n'.join(unknownWords)+'\n')
    return unknownWords


def create_training_testing_dataset_for_member_prediction(NodeChildrenFile="",NumOfSelect=100, NumOfChild=10, wsFile ="",
                                                          catPathFile="",  unknownWordsFile = "", lenUNK=100,
                                                          TrainPercentage=50, outPutFile=""):
    """
    :param NodeChildrenFile:
    :param NumOfSelect:
    :param NumOfChild:
    :param TrainPercentage:
    :param outPutFile:
    :return:
    """
    WSCHD = dict()
    outPutFile += "_".join([str(NumOfChild), str(TrainPercentage)])

    CATPATHDic = dict()
    with open(catPathFile, 'r') as pfh:
        for ln in pfh:
            wlst = ln[:-1].split()
            CATPATHDic[wlst[0]] = wlst[2:-1]

    wsLst = []
    with open(wsFile, 'r') as ifh:
        for ln in ifh:
            wsLst.append(ln[:-1])

    unkLst = []
    with open(unknownWordsFile, 'r') as ifh:
        for ln in ifh:
            word = ln[:-1]
            if '.' not in word and not word.isdigit():
                unkLst.append(ln[:-1])

    with open(NodeChildrenFile, 'r') as ifh:
        for ln in ifh:
            wlst = ln[:-1].split()
            if len(wlst[1:]) >= NumOfChild:
                WSCHD[wlst[0]] = wlst[1:]
    klst = list(WSCHD.keys())
    random.shuffle(klst)
    num = 0
    open(outPutFile, 'w').close()
    with open(outPutFile, 'a') as ofh:
        while num < NumOfSelect and klst:
            curKey = klst.pop()
            if curKey == "*root*": continue
            clst = [ele for ele in WSCHD[curKey] if ele in wsLst]
            if len(clst) < NumOfChild: continue

            EntitiesInKG = list(CATPATHDic.keys())
            allNodesInTree = CATPATHDic[curKey] + [curKey] + clst + ["*root*"]
            falseEntityInKG = list(set(EntitiesInKG).difference(allNodesInTree))

            random.shuffle(clst)
            boarder = math.ceil(len(clst) * TrainPercentage/100)
            trainLst = clst[:boarder]
            testLst = clst[boarder:]
            random.shuffle(falseEntityInKG)
            random.shuffle(unkLst)
            tRecord = curKey+"#"+" ".join(trainLst)+"#"+" ".join(testLst)+"#"+\
                      " ".join(falseEntityInKG[:lenUNK])+"#"+" ".join(unkLst[:lenUNK])
            ofh.write(tRecord+'\n')
            num += 1
    print(outPutFile, ' is created for training and testing\n')
    return outPutFile
