import numpy as np
import decimal
from nball4tree.config import DECIMAL_PRECISION
from nball4tree.util_train import get_children
from nball4tree.util_file import create_ball_file
from nball4tree.util_vec import vec_cos, vec_norm, vec_point, qsr_DC_degree, qsr_P_degree, rotate

decimal.getcontext().prec = DECIMAL_PRECISION


def shift_whole_tree_of(tree, deltaVec, deltaL, wsChildrenDic=dict(), word2ballDic=dict(), outputPath=None):
    """
    :param tree:
    :param deltaVec:
    :param deltaL:
    :param wsChildrenDic:
    :param word2ballDic:
    :param outputPath:
    :return:


    for child of tree:
        shift_whole_tree_of(child, deltaVec, deltaL, outputPath=None)

    l1, r1 = word2ballDic[tree][-2:]
    l = np.sqrt(l1*l1 + deltaL*deltaL
                    + 2*l1*deltaL* vec_cos(deltaVec, word2ballDic[tree][:-2]))
    newVec = vec_norm(vec_point(word2ballDic[tree][:-2], l1) + vec_point(deltaVec, deltaL))
    word2ballDic[tree] = list(newVec) + [l, r1]

    for child of tree:
        while True:
            delta = qsr_DC_degree_by_name(child, tree)
            if delta < 0:
                word2ballDic[tree][-2] += - delta*1.01
            else:
                break

    create_ball_file(tree, outputPath=outputPath)
    """
    for child in get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic):
        word2ballDic = shift_whole_tree_of(child, deltaVec, deltaL, wsChildrenDic=wsChildrenDic,
                                           word2ballDic=word2ballDic, outputPath=outputPath)

    l1, r1 = word2ballDic[tree][-2:]
    l = np.sqrt(l1 * l1 + deltaL * deltaL
                + 2 * l1 * deltaL * vec_cos(deltaVec, word2ballDic[tree][:-2]))
    newVec = vec_norm(vec_point(word2ballDic[tree][:-2], l1) + vec_point(deltaVec, deltaL))
    word2ballDic[tree] = list(newVec) + [l, r1]

    i, j, lst = 0, 0, get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic)
    for i in range(len(lst) - 1):
        j = i + 1
        while j < len(lst):
            dcDelta = qsr_DC_degree(word2ballDic[lst[i]], word2ballDic[lst[j]])
            if dcDelta < 0:
                print(lst[j],lst[i], j, i)
                word2ballDic = rotate_vector_till(lst[j],lst[i], word2ballDic =word2ballDic, logFile='word2ball.log')
            j += 1

    for child in get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic):
        gap = 1
        while True:
            delta = qsr_P_degree(word2ballDic[child], word2ballDic[tree])
            if delta < 0:
                gap *= 2
                word2ballDic[tree][-1] += - delta + gap
            else:
                break
    if outputPath:
        create_ball_file(tree, outputPath=outputPath, word2ballDic=word2ballDic)
    return word2ballDic


def rotate_vector_till(vec, vecRef, word2ballDic=dict(), logFile=None):
    """

    :param vec:
    :param vecRef:
    :param word2ballDic:
    :param logFile:
    :return:
    """

    dcDelta = qsr_DC_degree(word2ballDic[vecRef], word2ballDic[vec])
    if dcDelta < 0:
        rotateFlag = True
    rot1 = " ".join(["rotate from ", str(vec)] +
                    [str(ele) for ele in word2ballDic[vec][:-1]])
    k = 0
    while dcDelta < 0:
        l1, l2 = word2ballDic[vec][-2], word2ballDic[vec][-2]
        alpha = (l1 * l1 + l2 * l2 - dcDelta * dcDelta) / l1 / l2 / 2
        word2ballDic[vec][:-2] = rotate(word2ballDic[vec][:-2], alpha)
        print('sh in rotation alpha ', k, alpha)
        k += 1
        dcDelta = qsr_DC_degree(word2ballDic[vecRef], word2ballDic[vec])
    if rotateFlag and logFile is not None:
        with open(logFile, 'a+') as wlog:
            wlog.write(rot1 + "\n")
            wlog.write(" ".join(["rotate to ", str(vec)] +
                                [str(ele) for ele in word2ballDic[vec][:-1]]) + "\n")
    return word2ballDic


def ratio_homothetic_DC_transform(curTree, refTree, wsChildrenDic=dict(),
                                          word2ballDic=dict(), outputPath=None, logFile=None):
    """
     update curTree and all its children, that that they disconnect from refTree
    step 1 compute the ratio
        curTree central point P1, l1=|OP1|, radius r1, k = r1/l1
        refTree cnetral point P0, l0=|OP0|, radius r0
        (r0 + k*x)^2 = l0^2 + x^2 - 2*l0*x*cos\alpha
        x < (l0 + r0)/(1 - k) on the same line
    step 2 update the family of curTree

    :param curTree:
    :param refTree:
    :param wsChildrenDic:
    :param word2ballDic:
    :param outputPath:
    :param logFile:
    :return:
    """
    ball1 = word2ballDic[curTree]
    l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
    ball0 = word2ballDic[refTree]
    l0, r0 = decimal.Decimal(ball0[-2]), decimal.Decimal(ball0[-1])
    k = r1/l1
    targetsin0 = 0.6
    while k >= 1:

        print("assertion -1 k=", k)
        L, R = word2ballDic[curTree][-2:]

        print('Shifting...', curTree)
        LNew = R / decimal.Decimal(targetsin0)
        with open(logFile, 'a+') as wlog:
            wlog.write(" ".join(["shifting", str(curTree)] +
                                [str(ele) for ele in word2ballDic[curTree][:-2]] + [str(LNew - L)]))
            wlog.write("\n")
        word2ballDic = shift_whole_tree_of(curTree, word2ballDic[curTree][:-2], LNew - L,
                                           wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic,
                                           outputPath=outputPath)
        print('Ended of shifting...', curTree)

        ball1 = word2ballDic[curTree]
        l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
        k = r1 / l1
        targetsin0 *= 0.9

    margin = 10
    while True:
        assert word2ballDic[curTree][-2] != np.inf and word2ballDic[curTree][-2] >= 0

        ratio = decimal.Decimal(margin + l0 + r0)/decimal.Decimal(word2ballDic[curTree][-2]-word2ballDic[curTree][-1])
        l = word2ballDic[curTree][-2]
        word2ballDic[curTree][-2] = l * ratio
        word2ballDic[curTree][-1] = l * ratio - (l - word2ballDic[curTree][-1]) * ratio
        delta = qsr_DC_degree(word2ballDic[curTree],word2ballDic[refTree])
        if delta > 0:
            break
        decimal.getcontext().prec +=10
        margin *= 10
    if outputPath:
        create_ball_file(curTree, outputPath=outputPath, word2ballDic=word2ballDic)
    with open(logFile, 'a+') as wlog:
        wlog.write(" ".join(["homo", str(curTree)] + [str(ratio)]))
        wlog.write("\n")
    return ratio, word2ballDic


def homothetic_recursive_transform_of_decendents(tree, root=None, rate=None,
                                                         wsChildrenDic=dict(), word2ballDic=dict(), outputPath=None):
    """
        for child of tree:
        homothetic_recursive_transform_of_decendents_by_name(child, rate=None, outputPath=outputPath)
    l1, r1 = word2ballDic[tree][-2:]
    l = np.sqrt(l1*l1 + deltaL*deltaL
                    + 2*l1*deltaL* vec_cos(deltaVec, word2ballDic[tree][:-2]))
    newVec = vec_norm(vec_point(word2ballDic[tree][:-2], l1) + vec_point(deltaVec, deltaL))
    word2ballDic[tree] = list(newVec) + [l, r1]

    for child of tree:
        while True:
            delta = qsr_DC_degree_by_name(child, tree)
            if delta < 0:
                word2ballDic[tree][-2] += - delta*1.01
            else:
                break

    create_ball_file(tree, outputPath=outputPath)

    :param tree:
    :param root:
    :param rate:
    :param wsChildrenDic:
    :param word2ballDic:
    :param outputPath:
    :return:
    """
    if rate != 1:
        for child in get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic =word2ballDic):
            word2ballDic = homothetic_recursive_transform_of_decendents(child, root=root, rate=rate,
                                                                        word2ballDic =word2ballDic,
                                                                        wsChildrenDic=wsChildrenDic,
                                                                        outputPath=outputPath)

        if tree == root:
            return word2ballDic

        l = decimal.Decimal(word2ballDic[tree][-2])
        # l = word2ballDic[tree][-2]
        word2ballDic[tree][-2] = l * rate

        assert word2ballDic[tree][-2] != np.inf and word2ballDic[tree][-2] >= 0

        word2ballDic[tree][-1] = l * rate - (l - word2ballDic[tree][-1]) * rate
        # word2ballDic[tree][-1] *=  rate
        if outputPath:
            create_ball_file(tree, outputPath=outputPath, word2ballDic=word2ballDic)

        i, j, lst = 0, 0, get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic)
        for i in range(len(lst) - 1):
            j = i + 1
            while j < len(lst):
                dcDelta = qsr_DC_degree(word2ballDic[lst[i]], word2ballDic[lst[j]])
                if dcDelta < 0:
                    print(lst[j], lst[i], j, i)
                    word2ballDic=rotate_vector_till(lst[j], lst[i], word2ballDic=word2ballDic, logFile='word2ball.log')
                j += 1

        if outputPath:
            for child in get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic):
                create_ball_file(child,  outputPath=outputPath, word2ballDic=word2ballDic)

        for child in get_children(tree, wsChildrenDic=wsChildrenDic, word2ballDic=word2ballDic):
            gap = 1
            while True:
                delta = qsr_P_degree(word2ballDic[child], word2ballDic[tree])
                if delta < 0:
                    print('delta:', delta)
                    word2ballDic[tree][-1] += - delta + gap
                    gap *= 10
                else:
                    break
        if outputPath:
            create_ball_file(tree, outputPath=outputPath, word2ballDic=word2ballDic)
    return word2ballDic