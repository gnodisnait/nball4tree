
import statistics
import decimal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nball4tree.config import DECIMAL_PRECISION

decimal.getcontext().prec = DECIMAL_PRECISION


def show_membership_prediction_result(filePat="/Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt",
                                      numOfChild = 10,
                                      pers=[]):
    """
    :param filePat:
    :param pers:
    :return:
    """
    ttps, tfps, tfns = [], [], []
    t_ps, t_rs, t_f1 = [], [], []
    trmode, trmean, trpstdv = [], [], []
    tpmode, tpmean, tppstdv = [], [], []
    for per in pers:
        fileName = filePat + str(numOfChild)+ "_" + str(per)
        filePath, _ = os.path.split(filePat)
        with open(os.path.join(filePath, fileName), 'r') as ifh:
            tps, fps, fns, ps, rs = [], [], [], [], []  # true positives, false positives, false negatives, precisions, recalls
            for ln in ifh:
                if not ln.startswith('#'): continue
                elst = ln[1:-1].split()
                tps.append(float(elst[0]))
                fps.append(float(elst[1]))
                fns.append(float(elst[2]))
                ps.append(float(elst[4]))
                rs.append(float(elst[6]))

            ttps.append(sum(tps))
            tfps.append(sum(fps))
            tfns.append(sum(fns))

            if (ttps[-1] + tfps[-1] == 0):
                t_ps.append(0)
            else:
                t_ps.append(ttps[-1] / (ttps[-1] + tfps[-1]))
            precision = t_ps[-1]
            recall = ttps[-1] / (ttps[-1] + tfns[-1])
            f1 = 2 * precision * recall / (precision + recall)
            t_rs.append(float(format(recall, '.3f')))
            t_f1.append(float(format(f1, '.3f')))

            trmode.append(float(format(statistics.mode(rs), '.1f')))
            trmean.append(float(format(statistics.mean(rs), '.3f')))
            trpstdv.append(float(format(statistics.pstdev(rs), '.3f')))

            tpmode.append(float(format(statistics.mode(ps), '.1f')))
            tpmean.append(float(format(statistics.mean(ps), '.3f')))
            tppstdv.append(float(format(statistics.pstdev(ps), '.3f')))

    print(pers)
    print(t_ps)
    print(t_rs)

    recall_patch = mpatches.Patch(color='green', label='recall')
    precision_patch = mpatches.Patch(color='blue', label='precision')
    f1_patch = mpatches.Patch(color='red', label='F1')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(pers, t_ps, '-bs', pers, t_rs, '-g^', pers, t_f1, '--ro')
    for i, j in zip(pers, t_ps):
        ax.annotate(str(j), xy=(i, j))
    for i, j in zip(pers, t_rs):
        ax.annotate(str(j), xy=(i, j))
    for i, j in zip(pers, t_f1):
        ax.annotate(str(j), xy=(i, j))
    ax.legend(handles=[precision_patch, recall_patch, f1_patch])
    plt.xticks(pers)

    plt.show()

    mode_patch = mpatches.Patch(color='green', label='mode')
    mean_patch = mpatches.Patch(color='blue', label='mean')
    pstdv_patch = mpatches.Patch(color='red', label='pstdev')

    print(trmean)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(pers, trmode, '-g^', pers, trmean, '-bs', pers, trpstdv, '--ro')
    for i, j in zip(pers, trmode):
        ax.annotate(str(j), xy=(i, j))
    for i, j in zip(pers, trmean):
        ax.annotate(str(j), xy=(i, j))
    for i, j in zip(pers, trpstdv):
        ax.annotate(str(j), xy=(i, j))
    ax.legend(handles=[mode_patch, mean_patch, pstdv_patch])
    plt.xticks(pers)
    plt.show()


def summarize_experiment_result_in_table(ipath="", ifile="", ofile="",
                                         cols=['hypernymy', 'A', 'B', 'Sample',
                                               'TP', 'FP', 'TN',
                                               'precision', 'recall'],
                                         sampleNum=1):
    """
    ifile structure every two lines an experiment result
    :param ifile:
    :param ofile:
    :param cols:
    :param sampleNum:
    :return:
    """
    lines = []#['&'.join(cols)+"\\\\\\hline"]
    with open(os.path.join(ipath, ifile), 'r') as ifh:
        i = 0
        for ln in ifh:
            if ln.startswith('in all'): continue
            i += 1
            if ln.startswith('#'):
                tpos, fpos, tneg = [int(ele) for ele in ln[1:].split()[:3]]
            else:
                wlst = ln[:-1].split('#')
                hypernymy, knownLst, unknownLst = wlst[0], wlst[1].split(), wlst[2].split()
            if i % 2 == 0:
                num = len(knownLst)+len(unknownLst)

                if tpos+fpos != 0:
                    rlst = [hypernymy, str(num), str(len(knownLst))] + unknownLst[:sampleNum]+\
                           [str(ele) for ele in [tpos, fpos, tneg, tpos/(tpos+fpos), round(tpos/(tpos+ tneg),2)]]
                else:
                    rlst = [hypernymy, str(num), str(len(knownLst))] + unknownLst[:sampleNum] + \
                           [str(ele) for ele in [tpos, fpos, tneg, '-', 0]]
                lines.append(rlst)
                print(rlst)


    lines = sorted(lines, key=lambda ele: ele[0])
    nlines = []
    for wlst in lines:
        if int(wlst[1]) >= 50 or wlst[0].split('.')[0].endswith("ist"):
            nlines.append(["{\\bf "+wlst[0]+"}"]+wlst[1:])
        else:
            nlines.append(wlst)
    lines = ['&'.join(ele)+"\\\\\\hline" for ele in nlines]
    with open(os.path.join(ipath, ofile), 'w') as ofh:
        ofh.write('\n'.join(lines))


def summarize_all_experiment_results_in_table(ipath="", precentLst=[],
                                              catSize = 100,
                                                ifile="",
                                                ofile="", sampleNum=1):
    lines = []  # ['&'.join(cols)+"\\\\\\hline"]
    for percent in precentLst:
        ifile0 =ifile+str(percent)
        with open(os.path.join(ipath, ifile0), 'r') as ifh:
            i = 0
            for ln in ifh:
                if ln.startswith('in all'): continue
                i += 1
                if ln.startswith('#'):
                    tpos, fpos, tneg = [int(ele) for ele in ln[1:].split()[:3]]
                else:
                    wlst = ln[:-1].split('#')
                    hypernymy, knownLst, unknownLst = wlst[0], wlst[1].split(), wlst[2].split()
                if i % 2 == 0:
                    num = len(knownLst) + len(unknownLst)

                    if tpos + fpos != 0:
                        rlst = [hypernymy, str(num), str(len(knownLst))] + unknownLst[:sampleNum] + \
                               [str(ele) for ele in
                                [tpos, fpos, tneg, tpos / (tpos + fpos), round(tpos / (tpos + tneg), 2)]]
                    else:
                        rlst = [hypernymy, str(num), str(len(knownLst))] + unknownLst[:sampleNum] + \
                               [str(ele) for ele in [tpos, fpos, tneg, '-', 0]]
                    if num >= catSize:
                        lines.append(rlst)
                        print(rlst)

    lines = sorted(lines, key=lambda ele: ele[0])
    lines = ['&'.join(ele) + "\\\\\\hline" for ele in lines]
    with open(os.path.join(ipath, ofile), 'w') as ofh:
        ofh.write('\n'.join(lines))


def get_recall_from_files(trainingRatioLst, marginLst, target, ipath="", ifilePat="", otableFile=""):
    tmMatrix = []
    lines = []
    for r in trainingRatioLst:
        recallLst = []
        ln = [str(r)]
        for m in marginLst:
            fname = ifilePat.format(str(r), str(m))
            with open(os.path.join(ipath, fname), 'r') as ifh:
                lastLine = ifh.readlines()[-1]
                if lastLine.startswith("in all:"):
                    sval = lastLine.split('{}:'.format(target))[-1].split()[0]
                    print('*{}*'.format(sval))
                    recallLst.append(float(sval))
                    ln.append(str(sval)[:4])
        tmMatrix.append(recallLst)
        lines.append('&'.join(ln)+"\\\hline")
    with open(os.path.join(ipath, otableFile), 'w') as ofh:
        ofh.write('\n'.join(lines))
    tmMatrix = np.matrix(tmMatrix).transpose()
    return tmMatrix

