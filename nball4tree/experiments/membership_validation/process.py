"""
question: is x a member of class y? eg. is dog.n.01 a member of animal.n.01?

1. we have word embedding of dog, animal.
2. we know some members of animal.n.01, e.g. cat.n.01, horse.n.01
3. we know all the upper categories of animal.n.01.
4 (optional). we know the location of animal.n.01 in the taxonomy structure.

=======create training/testing datasets============
1. select nodes which have at least N children
2. from each node select p% children as known memebers, the rest children as testing data, create evaluation record
3. shuffle all the word-senses, choose the first K members who have at least N children,
    create a record: cat # p% of cat's children for training # 1-p% of cat's chidlren for testing
4. all above records are saved in a file catPredict.txt
"""

import decimal
import time
import os
import numpy as np
from nball4tree.config import DECIMAL_PRECISION
from collections import defaultdict
from nball4tree.main_training_process import training_one_family
from nball4tree.util_vec import qsr_P, vec_norm, vec_point, vec_cos, rotate

decimal.getcontext().prec = DECIMAL_PRECISION

'''
def training_one_family(treeStruc=defaultdict(list), root=None, w2vDic=dict(), catMemDic=dict(), catFPDic=dict(),
                        addDim=[], ballDict=dict(), L0=0, R0=0, logFile=""):
    children = treeStruc[root]
    if root in children:
        print('break')
    if len(children) > 0:
        for child in children:
            ballDict = training_one_family(treeStruc=treeStruc, root=child, w2vDic=w2vDic, catFPDic=catFPDic,
                                           catMemDic=catMemDic, addDim=addDim, ballDict=ballDict, L0=L0, R0=R0,
                                           logFile=logFile)
            # print('1',len(ballDict), list(ballDict.keys()))
        # children shall be separated
        if len(children) > 1:
            # print('training dc of root', root)
            ballDict = training_DC_by_name(children, word2ballDic=ballDict, wsChildrenDic=catMemDic, logFile=logFile)
            # print('2', len(ballDict), list(ballDict.keys()))
        # root ball shall contain all children balls
        ballDict = making_ball_contains(root, children, word2ballDic=ballDict, word2vecDic=w2vDic,
                                        wscatCodeDic=catFPDic,
                                        wsChildrenDic=catMemDic, addDim=addDim, logFile=logFile)
        # print('3', len(ballDict), list(ballDict.keys()))
        return ballDict
    else:
        _, ballDict = initialize_ball(root, word2vecDic=w2vDic, word2ballDic=ballDict, wscatCodeDic=catFPDic,
                                   addDim=addDim, L0=L0, R0=R0)

        # print('4', len(ballDict), list(ballDict.keys()))
        return ballDict
'''


def membership_prediction(trainingTestingFile="", outputFile="", NodeChildrenFile="", catPathFile="",
                          catFingerPrintFile="", w2vFile="", logPath="", L0=0, R0=0,  addDim=[]):
    """
    :param trainingTestingFile:
    :param outputFile:
    :return:
    """
    PCHDic = dict()
    with open(NodeChildrenFile, 'r') as ifh:
        for ln in ifh:
            wlst = ln[:-1].split()
            PCHDic[wlst[0]] = wlst[1:]

    CATPATHDic = dict()
    with open(catPathFile, 'r') as pfh:
        for ln in pfh:
            wlst = ln[:-1].split()
            CATPATHDic[wlst[0]] = wlst[2:-1]

    CATFPDic = dict()
    with open(catFingerPrintFile, 'r') as fp:
        for ln in fp:
            wlst = ln[:-1].split()
            CATFPDic[wlst[0]] = [int(ele) for ele in wlst[1:]]

    W2VDic = dict()
    with open(w2vFile, 'r') as efh:
        for ln in efh:
            wlst = ln[:-1].split()
            W2VDic[wlst[0]] = [decimal.Decimal(ele) for ele in wlst[1:]]

    tpTotal, fpTotal, tnTotal = 0, 0, 0

    flend = trainingTestingFile.split('.')[-1][3:]
    if flend:
        outputFile = outputFile + flend
    else:
        flend = '.'.join(trainingTestingFile.split('.')[-2:])
        flend =flend[3:]
        outputFile = outputFile + flend

    open(outputFile, 'w').close()

    with open(outputFile, 'a') as ofh:
        with open(trainingTestingFile, 'r') as ifh:
            for ln in ifh:
                wlst = ln[:-1].split('#')
                cat = wlst[0]
                trainingLst = wlst[1].split()
                testingLst = wlst[2].split()
                fLstInKG = wlst[3].split()
                UnkLst = wlst[4].split()
                logFile = "logFile_" +flend+ time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
                if not os.path.exists(logPath):
                    os.makedirs(logPath)
                open(os.path.join(logPath, logFile), 'w').close()

                tp,fp,tn = membership_prediction_evaluation(cat=cat, trainingLst=trainingLst, testingLst=testingLst,
                                                             falseListInKG=fLstInKG, UnknownEntities=UnkLst,
                                                             catMemDic = PCHDic,catPathDic = CATPATHDic,
                                                             catFPDic = CATFPDic, w2vDic = W2VDic,  L0=L0, R0=R0,
                                                             addDim = addDim,
                                                             logFile=os.path.join(logPath, logFile))
                tpTotal += tp
                fpTotal += fp
                tnTotal += tn
                if tp+fp == 0:
                    precision = 0 
                else:
                    precision = tp/(tp+fp)
                if tp+tn == 0:
                    recall = 0
                else:
                    recall = tp/(tp+tn)
                ofh.write(ln[:-1]+'\n#'+' '.join([str(ele) for ele in [tp, fp, tn, 'precision:', precision,
                                                                'recall:', recall]])+"\n")
                ofh.flush()
        precision = tpTotal/(tpTotal + fpTotal)
        recall = tpTotal/(tpTotal + tnTotal)
        ofh.write('in all:' + ' '.join([str(ele) for ele in ["precision:", precision, "recall:", recall]])+"\n")


def membership_prediction_evaluation(cat=None, trainingLst=[], testingLst=[],falseListInKG=[], UnknownEntities=[],
                                     catMemDic = None, catPathDic = None, catFPDic =None, w2vDic = None, L0=0, R0=0,
                                     addDim = [], logFile=""):
    """
    :param cat:
    :param trainingLst:
    :param testingLst:
    :param catPathDic:
    :param catFPDic:
    :param w2vDic:
    :return:
    """
    trainingTree = create_training_tree(cat=cat, children=trainingLst,catPathDic = catPathDic)

    ballDict = training_one_family(treeStruc=trainingTree, root=trainingTree['*root*'][0], word2ballDic=w2vDic,
                                    catMemDic=catMemDic, catFPDic=catFPDic, addDim = addDim, ballDict=dict(),
                                    L0=L0, R0=R0, logFile=logFile)

    tp, fp, tn = 0, 0, 0
    for tbname in testingLst:
        codeOfCat = catFPDic[tbname]
        print(codeOfCat)
        tball = create_testing_ball(tbname, cat, w2vDic=w2vDic, catPathDic=catPathDic, catFPDic=catFPDic,
                                    addDim=addDim, L0=L0, R0=R0, logFile=logFile)
        if qsr_P(tball, ballDict[cat]):
            tp += 1
        else:
            tn += 1

    # is incorrect word senses included by this cat ball?
    for bname in falseListInKG:
        tball = create_testing_ball(bname, cat, code=None,  w2vDic=w2vDic, catPathDic=catPathDic,
                                    catFPDic=catFPDic, addDim=addDim, logFile=logFile)
        if qsr_P(tball, ballDict[cat]):
            fp += 1

        tball = create_testing_ball(bname, cat, code=codeOfCat, w2vDic=w2vDic, catPathDic=catPathDic,
                                    catFPDic=catFPDic, addDim=addDim, logFile=logFile)
        if qsr_P(tball, ballDict[cat]):
            fp += 1

    for bname in UnknownEntities:
        print('validating unknown word ', bname, ' can be a ', cat, ' ?')
        tball = create_testing_ball(bname, cat, code=codeOfCat, w2vDic=w2vDic, catPathDic=catPathDic,
                                    catFPDic=catFPDic, addDim=addDim, logFile=logFile)
        if qsr_P(tball, ballDict[cat]):
            fp += 1
            print('fp +1')

    return tp, fp, tn


def create_training_tree(cat=None, children=[], catPathDic = None):
    ParentChildrenDic = defaultdict(list)
    plst = catPathDic[cat] +[cat]

    for p, ch in zip(plst[:-1], plst[1:]):
        ParentChildrenDic[p] = [ch]
    ParentChildrenDic[cat] = children
    if catPathDic[cat]:
        ParentChildrenDic['*root*'] = [catPathDic[cat][0]]
    else:
        ParentChildrenDic['*root*'] = [cat]
    return ParentChildrenDic


def create_testing_ball(bname, cat, code=None, w2vDic=None, catPathDic=None, catFPDic=None, addDim=[], L0=0, R0=0,
                        logFile=""):
    if code:
        fingerPrintUpCat = code
    else:
        fingerPrintUpCat = catFPDic[bname]

    if bname in w2vDic:
        wvec = w2vDic[bname]
    else:
        word = bname.split('.')[0]
        wvec = w2vDic[word]
    w2v = [decimal.Decimal(ele * 100) for ele in wvec]
    cpoint = w2v + [decimal.Decimal(ele) + 10 for ele in fingerPrintUpCat] + addDim
    tball = vec_norm(cpoint) + [L0, R0]
    tball = shitfing_htrans_one_testing_ball(tball, cat, catPathDic=catPathDic, logFile=logFile)
    return tball


def get_trans_history(logFile = None):
    """
    :param sFile:
    :return:
    """
    transh = []
    with open(logFile, 'r') as ifh:
        for ln in ifh:
            wlst = ln[:-1].split()
            if wlst[0] == "shifting":
                transh.append(['s', wlst[1], [decimal.Decimal(ele) for ele in wlst[2:]]])
            elif wlst[0] == "homo":
                transh.append(['h', wlst[1], decimal.Decimal(wlst[2])])
            elif wlst[0] == "rotate":
                rline = " ".join(wlst[2:])
                fromln, toln = rline.split('TO')
                transh.append(['r',
                               wlst[1],
                               [decimal.Decimal(ele) for ele in fromln.split()],
                               [decimal.Decimal(ele) for ele in toln.split()]])
    return transh


def shitfing_htrans_one_testing_ball(tball, parentName, catPathDic=None, logFile=""):
    """
    suppose parent is the parent node of the testChild, shift testChild according to the shifting history of parent
    :param tball :
    :param parent :
    :return:
    """
    ansLst = [parentName] + catPathDic[parentName]
    # cloc = copy.deepcopy(tball)
    transh = get_trans_history(logFile)
    for trans in transh:
        if trans[0] == 's' and trans[1] in ansLst:
            cvec, cL, cR = tball[:-2], tball[-2], tball[-1]
            vec, L = trans[2][:-1], trans[2][-1]
            npoint = vec_point(cvec, cL) + vec_point(vec, L)
            newVec = vec_norm(npoint)
            l1 = np.linalg.norm(npoint)
            tball[:-1] = newVec+ [l1]

        if trans[0] == 'h' and trans[1] in ansLst:
            ratio = trans[2]
            tball[-2] *= ratio
            tball[-1] *= ratio

        if transh[0] == 'r' and trans[1] in ansLst:
            rotatedCos = vec_cos(trans[2], trans[3])
            vecNew = rotate(tball, rotatedCos)
            tball = vecNew + tball[-2:]

    return tball


def membership_prediction_in_batch(plst, NumOfChild=10, trainingTestingFile="", outputFile="",
                              NodeChildrenFile="", catPathFile="",
                              catFingerPrintFile="", w2vFile="", logPath="",
                              L0= decimal.Decimal(1e+100), R0= decimal.Decimal(1e-200),
                              addDim = [512]*100):
    for trainPercentage in plst:
        taskFile = trainingTestingFile+ "_".join([str(NumOfChild), str(trainPercentage)])
        membership_prediction(trainingTestingFile=taskFile, outputFile=outputFile,
                              NodeChildrenFile=NodeChildrenFile, catPathFile=catPathFile,
                              catFingerPrintFile=catFingerPrintFile, w2vFile=w2vFile, logPath=logPath,
                              L0=L0, R0=R0, addDim=addDim)


'''
if __name__ == "__main__":
    word2vecFile = "/Users/tdong/data/glove/glove.6B.50d.txt"
    NodeChildrenFile = "/Users/tdong/data/glove/glove.6B.children.txt"
    membershipPredictionTask = "/Users/tdong/data/glove/memberValidation/membershipPredictionTask.txt"
    membershipPredictionResult = "/Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt"
    taxoPathFile = "/Users/tdong/data/glove/wordSensePath.txt"
    wordcatcodeFile = "/Users/tdong/data/glove/glove.6B.catcode.txt"
    logPath = "/Users/tdong/data/glove/logMemberValidate"

    precentLst = [5,10,20,30,40,50,60,70, 80, 90]

    membership_prediction_in_batch(precentLst,  NumOfChild=10, trainingTestingFile=membershipPredictionTask,
                                   outputFile=membershipPredictionResult, NodeChildrenFile=NodeChildrenFile,
                                   catPathFile=taxoPathFile, catFingerPrintFile=wordcatcodeFile,
                                   w2vFile=word2vecFile, logPath=logPath,
                                   L0= decimal.Decimal(1e+100), R0= decimal.Decimal(1e-200),
                                   addDim = [512]*100)
'''
    # show_trainingRatio_margin_relations(trainingRatioLst=precentLst, marginLst=ballMarginLst,
    #                                    target='precision',
    #                                    ipath="/Users/tdong/data/glove/glove.6B/",
    #                                    ifilePat="membershipPredictionResult.txt_Mushroom10_{}_MG{}",
    #                                    otableFile="marginTrainPercentTable"
    #                                    )


    # for i in precentLst:
    #    summarize_experiment_result_in_table(ipath="/Users/tdong/data/glove/glove.6B",
    #                                         ifile="newTest/membershipPredictionResult.txt10_"+str(i)+"_MG1",
    #                                         ofile="newTest/membershipPredictionResult.txt10_"+str(i)+"_MG1.table")

    # show_membership_prediction_result()
    # counting_triple_classification_datasets(precentLst)