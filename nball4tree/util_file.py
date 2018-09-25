import os
import decimal
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nball4tree.config import DECIMAL_PRECISION
from nball4tree.util_vec import vec_norm

decimal.getcontext().prec = DECIMAL_PRECISION


def create_ball_file(ballname, word2ballDic=dict(), outputPath=None):
    """

    :param ballname:
    :param word2ballDic:
    :param outputPath:
    :return:
    """
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    with open(os.path.join(outputPath, ballname), 'w+') as bfh:
        blst = word2ballDic[ballname]
        bfh.write(' '.join([str(ele) for ele in blst]) + "\n")


def load_one_ball(ball, ipath="/Users/tdong/data/glove/glove.6B/glove.6B.50Xball", word2ballDic=dict()):
    """
    :param ball:
    :param ipath:
    :param word2ballDic:
    :return:
    """
    with open(os.path.join(ipath, ball), 'r') as ifh:
        wlst = ifh.readline()[:-1].split()
        word2ballDic[ball] = [decimal.Decimal(ele) for ele in wlst]
        return len(wlst), 0, word2ballDic


def load_balls(ipath="/Users/tdong/data/glove/glove.6B/glove.6B.50Xball", word2ballDic=dict()):
    """

    :param ipath:
    :param word2ballDic:
    :return:
    """
    print("loading balls....")
    sizes = []
    dims = []
    for fn in [ele for ele in os.listdir(ipath) if len(ele.split('.'))>2]:
        sz, mele, word2ballDic = load_one_ball(fn, ipath = ipath, word2ballDic=word2ballDic)
        sizes.append(sz)
        dims.append(mele)
    sizes = list(set(sizes))
    print(sizes)
    print('totally', len(word2ballDic), ' balls are loaded')
    return max(sizes), min(dims), word2ballDic


def load_ball_embeddings(bFile):
    """
    :param bFile:
    :return:
    """
    print("loading balls....")
    bdic=dict()
    with open(bFile, 'r') as w2v:
        for line in w2v.readlines():
            wlst = line.strip().split()
            bdic[wlst[0]] = [decimal.Decimal(ele) for ele in wlst[1:]]
    print(len(bdic),' balls are loaded\n')
    return bdic


def get_word_embedding_dic(wordEmbeddingFile):
    dic=defaultdict()
    with open (wordEmbeddingFile, 'r') as fh:
        for line in fh.readlines():
            lst = line.split()
            v = [decimal.Decimal(ele) for ele in lst[1:]]
            dic[lst[0]] = v
    return dic


def get_ball_from_file(ball, ballPath = None):
    """

    :param ball:
    :param ballPath:
    :return:
    """
    with open(os.path.join(ballPath, ball), 'r') as ifh:
        wlst = ifh.readline().strip().split()
        ballv = [decimal.Decimal(ele) for ele in wlst]
    return ballv


def get_word_embedding_dic(wordEmbeddingFile):
    """

    :param wordEmbeddingFile:
    :return:
    """
    dic=dict()
    with open (wordEmbeddingFile, 'r') as fh:
        for line in fh.readlines():
            lst = line.split()
            v = [decimal.Decimal(ele) for ele in lst[1:]]
            dic[lst[0]] = v
    return dic


def initialize_dictionaries(word2vecFile=None, catDicFile=None, wsChildrenFile = None):
    """
    input is pre-trained word2vec

    :param word2vecFile:
    :param catDicFile:
    :param wsChildrenFile:
    :return:
    """
    wscatCodeDic = dict()
    wsChildrenDic = dict()
    word2vecDic = dict()
    if not os.path.isfile(word2vecFile):
        print('file does not exist:', word2vecFile)
        return

    with open(word2vecFile, 'r') as w2v:
        for line in w2v.readlines():
            wlst = line.strip().split()
            word2vecDic[wlst[0]] = vec_norm([float(ele) for ele in wlst[1:]])

    if os.path.isfile(catDicFile):
        with open(catDicFile, 'r') as cfh:
            for ln in cfh.readlines():
                wlst = ln[:-1].split()
                wscatCodeDic[wlst[0]] = [int(ele) for ele in wlst[1:]]

    if os.path.isfile(wsChildrenFile):
        with open(wsChildrenFile, 'r') as chfh:
            for ln in chfh:
                wlst = ln[:-1].split()
                wsChildrenDic[wlst[0]] = wlst[1:]
    return wsChildrenDic, word2vecDic, wscatCodeDic


def merge_balls_into_file(ipath="", outfile=""):
    """

    :param ipath:
    :param outfile:
    :return:
    """
    lst = []
    for fn in os.listdir(ipath):
        if fn.startswith('.'): continue
        with open(os.path.join(ipath, fn), "r") as fh:
            ln = fh.read()
            lst.append(" ".join([fn,ln]))
    with open(outfile, 'w+') as oth:
        oth.writelines(lst)


def get_all_words(aFile):
    with open(aFile) as f:
        words = f.read().split()
    return words


def create_parent_children_file(ln, ofile="/Users/tdong/data/glove_wordSenseChildren.txt",
                                w2vile= "/Users/tdong/data/glove/glove.6B.50d.txt"):
    """
    the problem of this method is: a->b->c, but b is not in the w2v file, a and c are in the w2v.
    the relation between a->c is brocken
    :param ln:
    :param ofile:
    :param w2vile:
    :return:
    """
    lst = ln.split()
    lines = [" ".join(["*root*"] + lst + ["\n"])]
    with open(w2vile, 'r') as vfh:
        vocLst = [word.split()[0] for word in vfh.readlines()]
    while lst:
        parent = lst.pop(0)
        children = [ele.name() for ele in wn.synset(parent).hyponyms() if ele.name().split('.')[0] in vocLst]
        newLine = " ".join([parent] + children + ["\n"])
        lines.append(newLine)
        print(parent, "::", children)
        lst += children
    with open(ofile, 'w') as ofh:
        ofh.write("".join(lines))


def create_parent_children_file_from_path(ofile="/Users/tdong/data/glove_wordSenseChildren.txt.newest",
                                            w2vFile= "/Users/tdong/data/glove/glove.6B.50d.txt",
                                            wsPath="/Users/tdong/data/glove/wordSensePath.txt.new"):
    def find_parent_of(x, ancestor=None):
        for lst in [[ele.name() for ele in hlst] for hlst in wn.synset(x).hypernym_paths()]:
            if ancestor in lst:
                return lst[-2]
    voc = []
    with open(w2vFile, 'r') as vfh:
        for ln in vfh:
            voc.append(ln.split()[0])
    parentChildDic = defaultdict(list)
    rootLst = []
    with open(wsPath, 'r') as ifh:
        for ln in ifh:
            wlst = ln.strip().split()[2:]
            if len(wlst) == 0:
                continue
            if wlst[0] not in rootLst:
                rootLst.append(wlst[0])
            for p,c in zip(wlst[:-1], wlst[1:]):
                pOfC = find_parent_of(c, ancestor=p)
                if c not in parentChildDic[pOfC]:
                    if not pOfC:
                        if not c:
                            parentChildDic[c]=[]
                    elif c in [ele.name() for ele in wn.synset(pOfC).instance_hyponyms()]:
                        if c not in parentChildDic[p]:
                            parentChildDic[p] += [ele.name() for ele in wn.synset(pOfC).instance_hyponyms()
                                                    if ele.name().split(".")[0] in voc]

                    elif c in [ele.name() for ele in wn.synset(pOfC).hyponyms()]:
                        if c not in parentChildDic[p]:
                            parentChildDic[p] += [ele.name() for ele in wn.synset(pOfC).hyponyms()
                                                    if ele.name().split(".")[0] in voc]

                    elif c not in parentChildDic[p]:
                        if c.split(".")[0] in voc:
                            parentChildDic[p] += [c]

    parentChildDic["*root*"] = rootLst
    lines = []
    for k, v in parentChildDic.items():
        if type(k) != str:
            print("*", k, "*")
        lines.append(" ".join([str(k)] + v + ["\n"]))
    with open(ofile, 'w') as ofh:
        ofh.write("".join(lines))


def clean_parent_children_file(ifile="/Users/tdong/myDocs/glove.6B.tree.input/glove.6B.children.txt",
                              w2vFile= "/Users/tdong/data/glove/glove.6B.50d.txt",
                               ofile="/Users/tdong/data/glove_wordSenseChildren.txt"):
    lines = []
    with open(w2vFile, 'r') as ifh:
        vocLst = [ele.split()[0] for ele in ifh.readlines()]
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            wlst = ln.strip().split()
            if wlst[0] == "*root*" or len(wlst) == 1:
                lines.append(" ".join(wlst+["\n"]))
            else :
                children = [ele for ele in wlst[1:] if ele.split(".")[0] in vocLst]
                lines.append(" ".join([wlst[0]] + children + ["\n"]))
    with open(ofile, 'w') as ofh:
        ofh.write("".join(lines))


def ball_counter(ifile):
    lst = []
    with open(ifile,  'r') as ifh:
        for ln in ifh:
            nlst = [ele for ele in ln.strip().split() if ele not in lst and ele != "*root*"]
            lst += nlst
    print(len(lst))


def clean_wordsense_path(ifile="", w2vFile ="", ofile=""):
    lines = []
    with open(w2vFile, 'r') as ifh:
        vocLst = [ele.split()[0] for ele in ifh.readlines()]
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            wlst = ln.strip().split()
            if len(wlst) > 2:
                node = wlst[0]
                lsts = [[ele.name() for ele in lst if ele.name().split(".")[0] in vocLst]
                        for lst in wn.synset(node).hypernym_paths()]
                wspath = sorted(lsts, key = len)[-1]
                lines.append(" ".join(wlst[:2] + wspath+["\n"]))
            else:
                lines.append(ln)
    with open(ofile, 'w') as ofh:
        ofh.write("".join(lines))


def create_ws_path(ifile="/Users/tdong/data/glove_wordSenseChildren57287.txt.newest.clean",
                   oWsPathfile="/Users/tdong/data/glove_wordSensePath57287.txt.newest"):
    """
    load all word-senses and wsChild2ParentDic from ifile
    for each ws:
        x = ws
        path[ws] = [ws]
        while x:
            x = get_parent_of(x)
            path[ws].append(x)

    :param ifile:
    :param ofile:
    :return:
    """
    wsParentDic, wsPathDic = defaultdict(list), defaultdict(list)
    wsLst = []
    with open(ifile, 'r') as cfh:
        for ln in cfh.readlines():
            lst = ln.strip().split()
            newWS = [ele for ele in lst if ele not in wsLst]
            wsLst += newWS
            for c in lst[1:]:
                wsParentDic[c] = lst[0]

    for ws in wsLst:
        wsPathDic[ws] = [ws]
        x = ws
        lst = [x]
        while x :
            x = wsParentDic[x]
            """
            'restrain.v.01'  and 'inhibit.v.04' have a mutal hyponym relation in WordNet 3.0
            'inhibit.v.04' hypernym path does not contain restrain.v.01
            manually set 'inhibit.v.04' as the hypernym of 'restrain.v.01'
            """
            if x in lst:
                print(x)
                x = False
                break
            if x:
                lst.append(x)
                wsPathDic[ws].append(x)

    open(oWsPathfile, 'w').close()
    with open(oWsPathfile, 'a+') as ofh:
        for ws, apath in wsPathDic.items():
            apath.reverse()
            ofh.write(" ".join([ws]+apath+["\n"]))


def generate_ws_cat_codes(cpathFile = "",
                          childrenFile ="",
                          outFile="", depth=0):
    """
    :param cpathFile:
    :param childrenFile:
    :param outFile:
    :param depth:
    :return:
    """
    wsPathDic, wsChildrenDic = defaultdict(), defaultdict()
    with open(cpathFile, 'r') as cfh:
        for ln in cfh.readlines():
            lst = ln[:-1].split()
            wsPathDic[lst[0]] = lst[1:]
    with open(childrenFile, 'r') as chfh:
        for ln in chfh.readlines():
            lst = ln.strip().split()
            if len(lst) == 0:
                wsChildrenDic[lst[0]] = []
            else:
                wsChildrenDic[lst[0]] = lst[1:]
    ofh = open(outFile, 'a+')
    ml, nm = 0, ''
    for node, plst in wsPathDic.items():
        plst = plst[:-1]
        clst = ["1"]
        if ml < len(plst):
            ml = len(plst)
            nm = node
        for (parent, child) in zip(plst[:-1], plst[1:]):
            children = wsChildrenDic[parent]
            clst.append(str(children.index(child) +1))
        clst += ['0'] * (depth - len(clst))
        line = " ".join([node] + clst) + "\n"
        ofh.write(line)
    ofh.close()
    return nm, ml


def check_whether_tree(ifile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest",
                       ofile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest.clean",
                       oWsPathfile="/Users/tdong/data/glove_wordSensePath57285.txt.newest.clean",
                       oCatCodeFile="/Users/tdong/data/glove_wordSenseCatCode57285.txt.newest.clean"):
    def appear2times(a, alst):
        if a in alst:
            loc = alst.index(a)
            if p in alst[loc+1:]:
                return True
        return False

    wsDic = defaultdict()
    with open(ifile , 'r') as ifh:
        for ln in ifh:
            wlst = ln.strip().split()
            for ws in wlst:
                if ws in wsDic:
                    wsDic[ws] += 1
                else:
                    wsDic[ws] = 1

    pLst = [(k, v-2) for k, v in wsDic.items() if v > 2]
    with open(ifile, 'r') as ifh:
        lines = ifh.readlines()
    while pLst:
        p, num = pLst.pop()
        for lnIndex in range(len(lines)):
            wlst = lines[lnIndex].strip().split()
            while p in wlst and wlst[0] != p and num > 0:
                wIndex = wlst.index(p)
                del wlst[wIndex]
                lines = lines[:lnIndex] +[" ".join(wlst+["\n"])] + lines[lnIndex+1:]
                num -= 1

    pLst = [k for k, v in wsDic.items() if v >= 2] # case of 'depression.n.02', case of 'pemphigus.n.01'
    print(' depression.n.02 in list',  'depression.n.02' in pLst)
    print('pemphigus.n.01 in list', 'pemphigus.n.01' in pLst)
    while pLst:
        p = pLst.pop()
        asLeaf = False
        for lnIndex in range(len(lines)):
            wlst = lines[lnIndex].strip().split()
            if p in wlst and wlst[0] != p and not asLeaf and not appear2times(p, wlst):
                asLeaf = True
            elif (appear2times(p, wlst) and wlst[0] != p and not asLeaf) or (p in wlst and wlst[0] != p and asLeaf):
                wIndex = wlst.index(p)
                del wlst[wIndex]
                lines = lines[:lnIndex] + [" ".join(wlst + ["\n"])] + lines[lnIndex + 1:]
                break

    with open(ofile, 'w') as ofh:
        ofh.write("".join(lines))

    create_ws_path(ifile=ofile, oWsPathfile=oWsPathfile)

    a, b = generate_ws_cat_codes(cpathFile=oWsPathfile,
                                 childrenFile=ofile,
                                 outFile=oCatCodeFile,
                                 depth=17)


def get_all_decendents(node, ifile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest.clean"):
    wsDic = defaultdict()
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            wlst = ln.strip().split()
            wsDic[wlst[0]] = wlst[1:]
    rlt, lst =[], [node]
    while lst:
        rlt += lst
        lst = sum([wsDic[ele] for ele in lst if ele in wsDic], [])
    return rlt


def check_two_forests_connected(node1, node2, ifile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest.clean"):
    lst1 = get_all_decendents(node1, ifile=ifile)
    lst2 = get_all_decendents(node2, ifile=ifile)
    rlt = [ele for ele in lst1 if ele in lst2]
    print(rlt)
    return rlt


if __name__ == "__main__":
    # check_two_forests_connected('body.n.02', 'organization.n.01',
    #                            ifile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest.clean")
    check_whether_tree()
    # create_ws_path(ifile="/Users/tdong/data/glove_wordSenseChildren57285.txt.newest.clean",
    #               ofile="/Users/tdong/data/glove_wordSensePath57285.txt.newest.clean")
    #  create_wordsense_path_from_ws_children()

    # clean_wordsense_path(ifile="/Users/tdong/data/glove/wordSensePath.txt",
    #                     w2vFile = "/Users/tdong/data/glove/glove.6B.50d.txt",
    #                     ofile="/Users/tdong/data/glove/wordSensePath.txt.new")
    # create_parent_children_file_from_path()
    # clean_parent_children_file()
    # ball_counter("/Users/tdong/data/glove_wordSenseChildren.txt.newest")

