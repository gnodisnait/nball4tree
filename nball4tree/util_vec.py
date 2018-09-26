
import numpy as np
import decimal
import copy
from nball4tree.config import DECIMAL_PRECISION

decimal.getcontext().prec = DECIMAL_PRECISION


def vec_norm(v):
    """
    :param v:
    :return:
    """
    return list(np.divide(v, np.linalg.norm(v)))


def vec_point(v, l):
    """
    :param v: v is the unit vector
    :param l: l is the lenth
    :return:
    """
    v1 = [decimal.Decimal(ele) for ele in v]
    l1 = decimal.Decimal(l)
    return np.multiply(v1, l1)


def vec_cos(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """
    v1 = [decimal.Decimal(ele) for ele in v1]
    v2 = [decimal.Decimal(ele) for ele in v2]
    return np.dot(v1, v2)


def dis_between(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """
    return np.sqrt(np.dot(v1 - v2, v1 - v2))


def dis_between_ball_centers(ball1, ball2):
    if ball1 == ball2:
        return 0
    cos = np.dot([decimal.Decimal(ele) for ele in ball1[:-2]], [decimal.Decimal(ele) for ele in ball2[:-2]])
    d2 = decimal.Decimal(ball1[-2] * ball1[-2] + ball2[-2] * ball2[-2]) \
         - 2 * decimal.Decimal(ball1[-2]) * decimal.Decimal(ball2[-2]) * decimal.Decimal(cos)
    # v1 = np.multiply([decimal.Decimal(ele) for ele in ball1[:-2]], ball1[-2])
    # v2 = np.multiply([decimal.Decimal(ele) for ele in ball2[:-2]], ball2[-2])
    # return dis_between(v1, v2)
    if d2 <0:
        return 0
    return np.sqrt(d2)



def qsr_P_degree(ball1, ball2):
    """
    check whether ball1 is part of ball2

    :param ball1:
    :param ball2:
    :return:
    """
    dis = dis_between_ball_centers(ball1, ball2)
    return ball2[-1] - dis - ball1[-1]


def qsr_P(ball1, ball2):
    """
    check whether ball1 is part of ball2
    :param ball1:
    :param ball2:
    :return:
    """
    degree = qsr_P_degree(ball1, ball2)
    if degree < 0:
        return False
    return True


def qsr_DC(ball1, ball2):
    """
        check whether ball1 disconnects from ball2
        :return: boolean
    """
    degree = qsr_DC_degree(ball1, ball2)
    if degree < 0:
        return False
    return True


def qsr_DC_degree(ball1, ball2):

    """
    check whether ball1 disconnects from ball2

    :param ball1:
    :param ball2:
    :return:
    """
    dis = dis_between_ball_centers(ball1, ball2)
    return dis - ball1[-1]-ball2[-1]


def rotate(vec, cosine):
    """
    :param vec:
    :param cosine:
    :return:
    """
    i = 100
    while cosine >= 1:
        cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        i -= 1

    while True:
        sinV = 1 - cosine*cosine
        if sinV < 0:
            sinV = 0
        else:
            sinV = np.sqrt(sinV)
        i = -1
        while vec[i] == 0:
            i -= 1
        j = i - 1
        while vec[j] == 0:
            j -= 1

        vecI0, vecJ0 = vec[i], vec[j]

        vec[i] = cosine *vec[i] + sinV*vec[j]
        vec[j] = -sinV *vec[i] + cosine*vec[j]

        if vec[i] == vecI0 and vec[j] == vecJ0:
            i -= 1
            cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        else:
            break
    return vec


def rotate(vec, cosine):
    """
    :param vec:
    :param cosine:
    :return:
    """
    i = 100
    while cosine >= 1:
        cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        i -= 1

    while True:
        sinV = 1 - cosine*cosine
        if sinV < 0:
            sinV = 0
        else:
            sinV = np.sqrt(sinV)
        i = -1
        while vec[i] == 0:
            i -= 1
        j = i - 1
        while vec[j] == 0:
            j -= 1

        vecI0, vecJ0 = vec[i], vec[j]

        vec[i] = cosine *vec[i] + sinV*vec[j]
        vec[j] = -sinV *vec[i] + cosine*vec[j]

        if vec[i] == vecI0 and vec[j] == vecJ0:
            i -= 1
            cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        else:
            break
    return vec


def average_vector(vecLst):
    N = len(vecLst)
    return np.divide(np.sum(vecLst, axis=0), N)


def dis_between_norm_vec(vec1, vec2):
    cos = np.dot([decimal.Decimal(ele) for ele in vec1], [decimal.Decimal(ele) for ele in vec2])
    d2 = 2 - 2 * decimal.Decimal(cos)
    # v1 = np.multiply([decimal.Decimal(ele) for ele in ball1[:-2]], ball1[-2])
    # v2 = np.multiply([decimal.Decimal(ele) for ele in ball2[:-2]], ball2[-2])
    # return dis_between(v1, v2)
    if d2 <0:
        return 0
    return np.sqrt(d2)


def spearmanr(mlst1, mlst2):
    smlst1 = copy.deepcopy(mlst1)
    smlst2 = copy.deepcopy(mlst2)
    squareD = 0
    N = len(mlst1)
    for i in range(N):
        m1 = mlst1[i]
        m2 = mlst2[i]
        i1 = smlst1.index(m1)
        i2 = smlst2.index(m2)
        squareD += (i1-i2)**2
    spm = 1 - 6*squareD/N/(N**2-1)
    return spm


def save_to_file(dstruc, ofile=""):
    open(ofile, 'w').close()
    with open(ofile, 'a') as ofh:
        if type(dstruc) == dict:
            for k, v in dstruc.items():
                if type(v) == list:
                    ln = ' '.join([k]+v)+'\n'
                if type(v) in [str, int, float]:
                    ln = ' '.join([k, str(v)]) + '\n'
                ofh.write(ln)
        if type(dstruc) == list:
            for ele in dstruc:
                if type(ele) == list:
                    ln = ' '.join(ele)+'\n'
                    ofh.write(ln)
                if type(ele) in [str, int, float]:
                    ofh.write(str(ln))