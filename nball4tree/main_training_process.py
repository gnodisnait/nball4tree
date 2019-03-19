import os
import copy
import time
import decimal
import operator
import numpy as np
from distutils.dir_util import copy_tree
from nball4tree.config import cgap, L0, R0, DIM, DECIMAL_PRECISION
from nball4tree.util_train import get_children
from nball4tree.util_vec import vec_norm, qsr_DC, qsr_DC_degree, qsr_P, qsr_P_degree,  dis_between_ball_centers
from nball4tree.util_file import create_ball_file, load_balls, get_ball_from_file, merge_balls_into_file, \
    initialize_dictionaries
from nball4tree.geo_transformation import ratio_homothetic_DC_transform, homothetic_recursive_transform_of_decendents,\
    shift_whole_tree_of

decimal.getcontext().prec = DECIMAL_PRECISION


def get_word2vector(wordsense, word2vecDic = dict()):
    """
    :param wordsense:
    :param word2vecDic:
    :return:
    """
    wd = wordsense.split('.')[0]
    if wd in word2vecDic:
        return word2vecDic[wd]
    elif wordsense.split('.')[0] in word2vecDic:
        return word2vecDic[wordsense.split('.')[0]]


def initialize_ball(root, addDim=[], L0=0.1, R0=0.1,
                    word2vecDic=dict(), wscatCodeDic=dict(), word2ballDic=dict(), outputPath=None):
    """
    :param root:
    :param addDim:
    :param L0:
    :param R0:
    :param word2vecDic:
    :param wscatCodeDic:
    :param word2ballDic:
    :param outputPath:
    :return:
    """
    w2v = [decimal.Decimal(ele*100) for ele in get_word2vector(root, word2vecDic=word2vecDic)]
    cpoint = w2v + [ele+10 for ele in wscatCodeDic[root]]+ addDim
    word2ballDic[root] = vec_norm(cpoint) + [L0, R0]
    if outputPath:
        create_ball_file(root,  outputPath=outputPath,word2ballDic=word2ballDic)
    return word2ballDic[root], word2ballDic


def training_P_by_name(childName, atreeName, addDim=[], wsChildrenDic=dict(),word2vecDic=dict(), wscatCodeDic=dict(),
                       word2ballDic=dict(), sep='.', outputPath="", logFile=None):
    """
    :param childName:
    :param atreeName:
    :param addDim:
    :param wsChildrenDic:
    :param word2vecDic:
    :param wscatCodeDic:
    :param word2ballDic:
    :param sep:
    :param outputPath:
    :param logFile:
    :return:
    """

    if childName.split(sep)[0] == atreeName.split(sep)[0]:
        BallLeaf = word2ballDic[childName]
        BallParent, word2ballDic = initialize_ball(atreeName, addDim=addDim, L0=L0, R0=R0, word2vecDic=word2vecDic,
                                                   wscatCodeDic=wscatCodeDic, word2ballDic=word2ballDic,
                                                   outputPath=outputPath)
        LeafO, ParentO = BallLeaf[:-2], BallParent[:-2]
        LeafL, LeafR = BallLeaf[-2],BallLeaf[-1]
        ParentL, ParentR = LeafL + LeafR + cgap, LeafR + LeafR + cgap + cgap
        BallParent = ParentO + [ParentL, ParentR]
        word2ballDic.update({atreeName: BallParent})
    else:
        targetsin0 = 0.6
        while True:
            BallLeaf = word2ballDic[childName]
            BallParent, word2ballDic = initialize_ball(atreeName, addDim=addDim, L0=L0, R0=R0, word2vecDic=word2vecDic,
                                                       wscatCodeDic=wscatCodeDic, word2ballDic=word2ballDic,
                                                       outputPath=outputPath)
            LeafO, ParentO = [decimal.Decimal(ele) for ele in BallLeaf[:-2]], \
                             [decimal.Decimal(ele) for ele in BallParent[:-2]]
            LeafL, LeafR = BallLeaf[-2],BallLeaf[-1]
            sin_beta = BallLeaf[-1] / BallLeaf[-2]

            delta = 1 - sin_beta * sin_beta
            if delta < 0:
                delta = 0
            cos_beta = np.sqrt(delta)
            cos_alpha = np.dot(LeafO, ParentO) / np.linalg.norm(LeafO)/ np.linalg.norm(ParentO)

            delta = 1 - cos_alpha * cos_alpha
            if delta < 0:
                delta = 0
            sin_alpha = np.sqrt(delta)

            # begin alpha --> xalpha
            xalpha = sin_alpha/25
            yalpha = np.sqrt(1 - xalpha*xalpha)
            sin_xalpha = xalpha*cos_alpha + yalpha*sin_alpha
            delta = 1 - sin_xalpha * sin_xalpha
            if delta < 0: delta = 0
            cos_xalpha = np.sqrt(delta)

            sin_alpha = sin_xalpha
            cos_alpha = cos_xalpha
            # end

            dOO = LeafL * decimal.Decimal(cos_beta)

            cos_alpha_beta = (decimal.Decimal(cos_beta) * decimal.Decimal(cos_alpha)
                          - decimal.Decimal(sin_beta) * decimal.Decimal(sin_alpha))
            if cos_alpha_beta <=0:
                # shift_one_family(root=childName, targetsin = targetsin0,  outputPath=outputPath)
                L, R = word2ballDic[childName][-2:]
                print('Shifting...', childName)
                LNew = R / decimal.Decimal(targetsin0)
                with open(logFile, 'a+') as wlog:
                    wlog.write(" ".join(["shifting",str(childName)]+
                                         [str(ele) for ele in word2ballDic[childName][:-2]] + [str(LNew - L)]))
                    wlog.write("\n")
                word2ballDic=shift_whole_tree_of(childName,  word2ballDic[childName][:-2], LNew - L,
                                                 wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic,
                                                 outputPath=outputPath)
                # check_P_for_child_parent_in_one_family(childName, ballPath=outputPath)
                checkResult = check_DC_for_sibilings_in_one_family(childName)
                if checkResult:
                    print("check_DC_for_sibilings_in_one_family", childName, checkResult)
                targetsin0 *= 0.9
            else:
                break

        ParentL = dOO / cos_alpha_beta
        assert ParentL > 0 and ParentL != np.inf

        ParentR = ParentL * (decimal.Decimal(sin_alpha) * decimal.Decimal(cos_beta)
                             + decimal.Decimal(cos_alpha) * decimal.Decimal(sin_beta)) + decimal.Decimal(0.1)
        BallParent = ParentO + [ParentL, ParentR]
        word2ballDic.update({atreeName: BallParent})

    count = 0
    while qsr_P_degree(word2ballDic[childName], word2ballDic[atreeName]) < 0:
        oldParentR, delta = ParentR, 10
        ParentR += decimal.Decimal(2) - qsr_P_degree(word2ballDic[childName], word2ballDic[atreeName])
        while oldParentR == ParentR:
            ParentR += delta
            delta *= 10
        BallParent = ParentO + [ParentL, ParentR]
        word2ballDic.update({atreeName: BallParent})
        # print('*', qsr_P_degree_by_name(childName, atreeName))
        # print("**", qsr_P_by_name(childName, atreeName))
        count += 1
        # print('count', count)

    # assert qsr_P_by_name(childName, atreeName), childName+" - "+atreeName+": "+str(qsr_P_degree_by_name(childName, atreeName))
    if outputPath:
        create_ball_file(atreeName,  outputPath=outputPath,word2ballDic=word2ballDic)
    return BallParent, word2ballDic


def making_ball_contains(root, children,  addDim=[], word2vecDic=dict(),
                         wsChildrenDic=dict(), wscatCodeDic=dict(), word2ballDic=dict(),
                         outputPath=None, logFile=None):
    """
    :param root:
    :param children:
    :param addDim:
    :param wsChildrenDic:
    :param wscatCodeDic:
    :param word2ballDic:
    :param outputPath:
    :param logFile:
    :return:
    """
    maxL = -1
    flag = False
    while not flag:
        flag = True
        for childName in children:
            pBall, word2ballDic = training_P_by_name(childName, root,  addDim=addDim,
                                       wsChildrenDic=wsChildrenDic, word2vecDic=word2vecDic, wscatCodeDic=wscatCodeDic,
                                       word2ballDic=word2ballDic,
                                       outputPath=outputPath, logFile=logFile)
            assert pBall != -1
            if maxL == -1: # initialize maxL, minL_R
                maxL, minL_R = pBall[-2], decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if maxL < pBall[-2]:
                maxL = pBall[-2]
            delta = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if delta <=0:
                print('Shifting...mbc', root)
                with open(logFile, 'a+') as wlog:
                    wlog.write(" ".join(["shifting",str(root)]+
                                         [str(ele) for ele in word2ballDic[root][:-2]] + [str(-delta)]))
                    wlog.write("\n")
                word2ballDic = shift_whole_tree_of(root, word2ballDic[root][:-2], -delta,
                                    wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic,
                                    outputPath=outputPath)
                flag = False
                break
            elif decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1]) < minL_R:
                minL_R = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])

            word2ballDic[root] = word2ballDic[root][:-2] + [maxL, maxL - minL_R + cgap]
            if outputPath:
                create_ball_file(root,  outputPath=outputPath,word2ballDic=word2ballDic)
    return word2ballDic


def training_DC_by_name(childrenNames, wsChildrenDic=dict(), word2ballDic=dict(),
                        outputPath=None, ordered = False, logFile=None):
    """
    :param childrenNames:
    :param wsChildrenDic:
    :param word2ballDic:
    :param outputPath:
    :param maxsize:
    :param mindim:
    :param logFile:
    :return:
    """
    dic = dict()
    for tree in childrenNames:
        dic[tree] = word2ballDic[tree][-2]
    dic0 = copy.deepcopy(dic)

    if ordered:
        lst = [(node, word2ballDic[node]) for node in childrenNames]
    else:
        lst = [(item[0], word2ballDic[item[0]]) for item in sorted(dic.items(), key=operator.itemgetter(1))]

    i = 0
    if "herd.n.02" in childrenNames and "gathering.n.01" in childrenNames:
        print('break')
    while i < len(lst) - 1:
        # print('i:', i, ' in', len(lst))
        j = i + 1
        refTreeName = lst[i][0]
        while j < len(lst):
            curTreeName = lst[j][0]
            # print(curTreeName, refTreeName)
            targetsin0 = 0.6
            while not qsr_DC(word2ballDic[curTreeName], word2ballDic[refTreeName]):
                ball1 = word2ballDic[curTreeName]
                l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
                k = r1 / l1
                if k == 1:
                    L, R = word2ballDic[curTreeName][-2:]
                    print('Shifting...', curTreeName)
                    LNew = R / decimal.Decimal(targetsin0)
                    with open(logFile, 'a+') as wlog:
                        wlog.write(" ".join(["shifting", str(tree)] +
                                            [str(ele) for ele in word2ballDic[tree][:-2]] + [str(LNew - L)]))
                        wlog.write("\n")
                    word2ballDic= shift_whole_tree_of(tree, word2ballDic[curTreeName][:-2], LNew - L,
                                                      wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic,
                                                      outputPath=outputPath)
                    # check_P_for_child_parent_in_one_family(tree, ballPath=outputPath)
                    checkResult=check_DC_for_sibilings_in_one_family(tree)
                    if checkResult:
                        print("check_DC_for_sibilings_in_one_family", tree, checkResult)
                    targetsin0 *= 0.9

                ratio0, word2ballDic = ratio_homothetic_DC_transform(curTreeName, refTreeName,
                                                                             wsChildrenDic=wsChildrenDic,
                                                                             word2ballDic=word2ballDic,
                                                                             outputPath=outputPath,
                                                                             logFile=logFile)
                assert ratio0 != -1

            # assert qsr_DC_by_name(curTreeName, refTreeName, outputPath=outputPath)
            if outputPath:
                create_ball_file(curTreeName, outputPath=outputPath, word2ballDic=word2ballDic)
            j += 1
        for tree in childrenNames:
            dic[tree] = word2ballDic[tree][-2]
        lst = [(item[0], word2ballDic[item[0]]) for item in sorted(dic.items(), key=operator.itemgetter(1))]
        i += 1

    if "herd.n.02" in childrenNames and "gathering.n.01" in childrenNames:
        print('break')

    #####
    # homothetic transformation
    #####
    for child in childrenNames:
        ratio = word2ballDic[child][-2]/decimal.Decimal(dic0[child])
        word2ballDic = homothetic_recursive_transform_of_decendents(child, root=child, rate=ratio,
                                                                    wsChildrenDic=wsChildrenDic,
                                                                    word2ballDic=word2ballDic, outputPath=outputPath)
    return word2ballDic


def training_one_family(treeStruc=None,root=None, addDim=[], wsChildrenDic = dict(), word2vecDic=dict(),
                        wscatCodeDic=dict(),
                        word2ballDic = dict(), outputPath=None, logFile=None):
    """
    :param treeStruc:
    :param root:
    :param addDim:
    :param wsChildrenDic:
    :param word2vecDic:
    :param wscatCodeDic:
    :param word2ballDic:
    :param outputPath:
    :param logFile:
    :return:
    """
    if treeStruc:
        children = treeStruc[root]
    else:
        children = get_children(root, wsChildrenDic=wsChildrenDic)
    if len(children) > 0:
        for child in children:
            word2ballDic = training_one_family(treeStruc=treeStruc, root=child, addDim=addDim,
                                               wsChildrenDic=wsChildrenDic,
                                               word2vecDic=word2vecDic, wscatCodeDic=wscatCodeDic,
                                               word2ballDic=word2ballDic,
                                               outputPath=outputPath, logFile=logFile)
        # children shall be separated
        if len(children) > 1:
            # print('training dc of root', root)
            word2ballDic = training_DC_by_name(children, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic,
                                               outputPath=outputPath, logFile=logFile)
        # root ball shal contain all children balls
        word2ballDic = making_ball_contains(root, children,  addDim=addDim, word2vecDic=word2vecDic,
                                            wsChildrenDic=wsChildrenDic, wscatCodeDic=wscatCodeDic,
                                            word2ballDic =word2ballDic, outputPath=outputPath, logFile=logFile)
        return word2ballDic

    else:
        ball, word2ballDic = initialize_ball(root,  addDim=addDim, L0=L0, R0=R0,
                                             word2vecDic=word2vecDic, word2ballDic=word2ballDic,
                                             wscatCodeDic=wscatCodeDic,
                                             outputPath=outputPath)
        return word2ballDic


def check_P_for_child_parent_in_one_family(root=None, wsChildrenDic=dict(), word2ballDic=dict(), ballPath=""):
    """
    :param root:
    :param wsChildrenDic:
    :param word2ballDic:
    :param ballPath:
    :return:
    """
    lst = [root]
    while lst:
        parent = lst.pop()
        pBall = get_ball_from_file(parent, ballPath = ballPath) #word2ballDic[parent]
        children = get_children(parent, wsChildrenDic=wsChildrenDic)
        lst += children
        for child in children:
            chBall = get_ball_from_file(child, ballPath = ballPath) #word2ballDic[child]
            if not qsr_P(word2ballDic[child], word2ballDic[parent]):
                print(child, parent, 'violates condition 3')
                dis = dis_between_ball_centers(chBall, pBall)
                print('dis:', dis)
                print('r1', chBall[-1])
                print('R', pBall[-1])
                print('shall >=0', pBall[-1]- dis - chBall[-1])
                # assert qsr_P_by_name(child, parent), str(qsr_P_degree_by_name(child, parent))
                return [root]
    # print("passed checking ", root, ' for part of')
    return []


def check_DC_for_sibilings_in_one_family(root="*root*", wsChildrenDic=dict(), word2ballDic=dict()):
    """
    :param root:
    :param wsChildrenDic:
    :param word2ballDic:
    :return:
    """
    lst = [root]
    checkResult = []
    while lst:
        parent = lst.pop()
        children = get_children(parent, wsChildrenDic=wsChildrenDic)
        lst += children
        if len(children) <2:
            continue
        i,j = 0, 0
        while i < len(children):
            j = i + 1
            while j < len(children):
                if not qsr_DC(word2ballDic[children[i]], word2ballDic[children[j]]):
                    print(children[i], children[j], 'violates condition 4')
                    # print('shall >=0', str(qsr_DC_degree(word2ballDic[children[i]], word2ballDic[children[j]])))
                    # return [root]
                    checkResult.append((children[i], children[j]))
                j += 1
            i += 1
    return checkResult


def training_all_families(root="*root*", wsChildrenDic=dict(), word2vecDic=dict(), wscatCodeDic=dict(),
                          word2ballDic=dict(),
                          outputPath=None, logFile=None, checking = False):
    """
    :param root:
    :param wsChildrenDic:
    :param word2vecDic:
    :param wscatCodeDic:
    :param word2ballDic:
    :param outputPath:
    :param logFile:
    :param checking:
    :return:
    """
    global L0, DIM
    children = get_children(root, wsChildrenDic=wsChildrenDic)
    child0= 'entity.n.01'
    children = sorted(children, key=lambda ele: np.dot(get_word2vector(child0, word2vecDic=word2vecDic),
                                                       get_word2vector(ele, word2vecDic=word2vecDic)))
    print(children)
    N = int(np.ceil(np.log(len(children))))
    open(logFile, 'w+')
    while children:
        child = children.pop()
        k = 512
        addDim0 = list(bin(N))[2:][:DIM]
        if len(addDim0) < DIM:
            addDim0 += [0] * (DIM - len(addDim0))
        addDim = [int(ele) * 2 - 1 for ele in addDim0]
        addDim = [ele * k for ele in addDim]
        print("***", child)
        with open(logFile, 'a+') as wlog:
            wlog.write(" ".join([str(ele) for ele in [child]
                                    +addDim
                                    +[time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())]]))
            wlog.write("\n")
        word2ballDic = training_one_family(root=child, addDim=addDim, wsChildrenDic=wsChildrenDic,
                                           word2vecDic=word2vecDic, wscatCodeDic=wscatCodeDic,
                                           word2ballDic=word2ballDic,
                                           outputPath=outputPath, logFile=logFile)
        children = sorted(children, key=lambda ele: np.dot(get_word2vector(child, word2vecDic=word2vecDic),
                                                           get_word2vector(ele, word2vecDic=word2vecDic)))
    print("finished training of all families\n")

    if checking:
        print("checking each family\n")
        maxsize, mindim, word2ballDic = load_balls(ipath=outputPath, word2ballDic=word2ballDic)

        failed_P, failed_DC = [], []

        for child in get_children(root):
            failed_P += check_P_for_child_parent_in_one_family(child, word2ballDic =word2ballDic,
                                                               wsChildrenDic=wsChildrenDic, ballPath=outputPath)
            failed_DC += check_DC_for_sibilings_in_one_family(root=child, word2ballDic =word2ballDic,
                                                              wsChildrenDic=wsChildrenDic)
        print("failed families with P", failed_P)
        print("failed families with DC", failed_DC)
    return word2ballDic


def testing_whole_family(outputPath=None, wsChildrenDic=dict(), word2ballDic=dict(), outputBallFile=None):
    """
    :param outputPath:
    :param wsChildrenDic:
    :param word2ballDic:
    :param outputBallFile:
    :return:
    """
    print("checking whether the tree structure is perfectly encoded in nball embeddings...\n")
    failed_P, failed_DC = [], []
    maxsize, mindim, word2ballDic = load_balls(ipath = outputPath, word2ballDic=word2ballDic)

    for froot in get_children('*root*', wsChildrenDic=wsChildrenDic):
        failed_P += check_P_for_child_parent_in_one_family(froot,
                                                           wsChildrenDic=wsChildrenDic,
                                                           word2ballDic=word2ballDic,
                                                           ballPath=outputPath)

    failed_DC += check_DC_for_sibilings_in_one_family(root='*root*', wsChildrenDic=wsChildrenDic,
                                                      word2ballDic=word2ballDic)
    print("failed families with P", failed_P)
    print("failed families with DC", failed_DC)
    if failed_P == [] and failed_DC == []:
        print("the tree structure is perfectly encoded in nball embeddings.\n")
        print("generating nball embedding file...\n")
        merge_balls_into_file(ipath= outputPath, outfile=outputBallFile)
    else:
        print("the tree structure is NOT perfectly encoded in nball embeddings.\n")
        print("try again, or contact the author")


def fix_dim(maxsize, mindim, word2ballDic=dict(), bPath = '/Users/tdong/data/glove/glove.6B/glove.6B.50Xball',
            outputPath=""):
    """
    :param maxsize:
    :param mindim:
    :param word2ballDic:
    :param bPath:
    :return:
    """
    for bf in os.listdir(bPath):
        with open(os.path.join(bPath, bf), 'r') as ifh:
            wlst = ifh.readline().strip().split()
            ballv = [decimal.Decimal(ele) for ele in wlst]
            delta = maxsize - len(ballv)
            if delta > 0:
                assert len(wlst) < maxsize
                print(bf, len(wlst), ballv[-1])
                vec = vec_norm(ballv[:-2] + [decimal.Decimal(mindim)] * delta) + ballv[-2:]
                word2ballDic[bf] = vec
                if outputPath:
                    create_ball_file(bf, outputPath=bPath,word2ballDic=word2ballDic)
    return word2ballDic


def make_DC_for_first_level_children(root="*root*", firstChild = 'entity.n.01', wsChildrenDic=dict(),
                                     outputPath='', maxsize=0, mindim=0,  word2ballDic = dict(),
                                     logFile=None):
    """
    :param root:
    :param firstChild:
    :param wsChildrenDic:
    :param outputPath:
    :param maxsize:
    :param mindim:
    :param word2ballDic:
    :param logFile:
    :param checking:
    :return:
    """
    children = get_children(root, wsChildrenDic=wsChildrenDic)
    children.remove(firstChild)
    children.insert(0, firstChild)
    print('updating first level children...')
    word2ballDic = training_DC_by_name(children, outputPath=outputPath, wsChildrenDic=wsChildrenDic,
                                       word2ballDic =word2ballDic, ordered = True,
                                       logFile=logFile)
    return word2ballDic


def train_word2ball(root="",  outputPath = '', logFile='', wsChildrenDic=dict(),
                    word2ballDic=dict(), word2vecDic=dict(), outputPathBack = None,
                    wscatCodeDic=dict(), outputBallFile=None):
    """
    :param root:
    :param outputPath:
    :param logFile:
    :param wsChildrenDic:
    :param word2ballDic:
    :param word2vecDic:
    :param wscatCodeDic:
    :param outputBallFile:
    :param outputBallForestFile:
    :return:
    """
    training_all_families(root=root, wsChildrenDic=wsChildrenDic, word2vecDic=word2vecDic,
                                          wscatCodeDic=wscatCodeDic, word2ballDic=word2ballDic,
                                          outputPath=outputPath, logFile=logFile)
    if outputPathBack:
        copy_tree(outputPath, outputPathBack)
    maxsize, mindim , word2ballDic = load_balls(ipath=outputPath, word2ballDic=word2ballDic)
    fix_dim(maxsize, mindim, bPath=outputPath, outputPath=outputPath)
    make_DC_for_first_level_children(root=root, firstChild = 'entity.n.01', wsChildrenDic=wsChildrenDic,
                                                    word2ballDic=word2ballDic, outputPath=outputPath,
                                                    maxsize=maxsize, mindim=mindim, logFile=logFile)

    testing_whole_family(outputPath=outputPath, wsChildrenDic=wsChildrenDic, outputBallFile=outputBallFile)

