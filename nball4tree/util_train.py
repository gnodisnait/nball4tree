import decimal
from nball4tree.config import DECIMAL_PRECISION

decimal.getcontext().prec = DECIMAL_PRECISION


def get_children(nodeData, wsChildrenDic=dict(), firstChild=None, word2ballDic=None):
    """
    get all children of a node
    if firstChild is given, order the children list

    :param nodeData:
    :param wsChildrenDic:
    :param firstChild:
    :return:
    """
    if word2ballDic:
        chlst = [ch for ch in wsChildrenDic.get(nodeData, []) if ch in word2ballDic]
    else:
        chlst = wsChildrenDic.get(nodeData, [])
    if firstChild in chlst:
        chlst.remove(firstChild)
        chlst.insert(0, firstChild)
    # add blacklist
    # chlst = [ele for ele in chlst if ele not in ['charisma.n.01']]
    return chlst