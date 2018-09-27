import os
import argparse
import decimal
from nball4tree import train_word2ball, initialize_dictionaries, testing_whole_family
from nball4tree.util_file import load_ball_embeddings
from nball4tree.experiments.nearest_neighbors.process import nearest_neighbors_of_word_sense
from nball4tree.experiments.membership_validation import membership_prediction_in_batch
from nball4tree.experiments.membership_validation import show_membership_prediction_result
from nball4tree.experiments.consistency import maximum_deviation
from nball4tree.config import DECIMAL_PRECISION
decimal.getcontext().prec = DECIMAL_PRECISION


"""
# usage 1: train and evaluate nball embeddings
# usage 1.1: train nball embeddings
$ python nball.py --train_nball /Users/tdong/data/glove/nball.txt --w2v /Users/tdong/data/glove/glove.6B.50d.txt   --ws_child /Users/tdong/data/glove/wordSenseChildren.txt    --ws_catcode /Users/tdong/data/glove/glove.6B.catcode.txt  --log log.txt
# usage 1.2:  evaluate nball embeddings
$ python nball.py --zero_energy <output-path> --ball <output-file> --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt 

# usage 2: list neighbors of a word or word-sense
$ python nball.py --neighbors beijing.n.01 berlin.n.01  --ball /Users/tdong/data/glove/glove.6B.50Xball.V10.txt  --num 6  

# usage 3: validate membership
$ python nball.py  --validate_member /Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt --numOfChild 10  --percentages 5 10 20 30 40 50 60 70 80 90  --taskFiles /Users/tdong/data/glove/memberValidation/membershipPredictionTask.txt --w2v /Users/tdong/data/glove/glove.6B.50d.txt --ws_child /Users/tdong/data/glove/wordSenseChildren.txt  --ws_path /Users/tdong/data/glove/wordSensePath.txt --ws_catcode /Users/tdong/data/glove/glove.6B.catcode.txt --logPath /Users/tdong/data/glove/logMemberValidate

# usage 4: plot result of validate membership
$ python nball.py  --plot_validate_member /Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt --numOfChild 10  --percentages 5 10 20 30 40 50 60 70 80 90 

# usage 5: consistency analysis 1: deviation of word-stems
$ python nball.py  --std_stem /Users/tdong/data/glove/wordstem.std --dim 50 --ballStemFile /Users/tdong/data/glove/glove.6B.50Xball.words --ball /Users/tdong/data/glove/glove.6B.50Xball.V10.txt
 """


def main():
    parser = argparse.ArgumentParser()

    """
    sample command for training nball embeddings
    $ python nball.py --train_nball /Users/tdong/data/glove/nball.txt --w2v /Users/tdong/data/glove/glove.6B.50d.txt \
        --ws_child /Users/tdong/data/glove/wordSenseChildren.txt   \
        --ws_catcode /Users/tdong/data/glove/glove.6B.catcode.txt  \
        --log log.txt
    """
    parser.add_argument('--train_nball')
    parser.add_argument('--w2v')
    parser.add_argument('--ws_child')
    parser.add_argument('--ws_catcode')
    parser.add_argument('--log')

    """
    evaluate nball embeddings
    $ python nball.py --zero_energy /Users/tdong/data/glove/data_out     --ball ball.txt   --ws_child /Users/tdong/data/glove/wordSenseChildren.txt  
    """
    parser.add_argument('--zero_energy')

    """
    sample command for listing neighbors of word-sense
    $ python nball.py --neighbors beijing.n.01 berlin.n.01  --ball /Users/tdong/data/glove/glove.6B.50Xball.V10.txt  --num 6  
    """
    parser.add_argument('--neighbors', nargs='*')
    parser.add_argument('--ball')
    parser.add_argument('--num', type=int)

    """
    # usage 3: validate membership
    $ python nball.py  --validate_member /Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt --numOfChild 10  --percentages 5 10 20 30 40 50 60 70 80 90  --taskFiles /Users/tdong/data/glove/memberValidation/membershipPredictionTask.txt --w2v /Users/tdong/data/glove/glove.6B.50d.txt --ws_child /Users/tdong/data/glove/wordSenseChildren.txt  --ws_path /Users/tdong/data/glove/wordSensePath.txt --ws_catcode /Users/tdong/data/glove/glove.6B.catcode.txt --logPath /Users/tdong/data/glove/logMemberValidate
    """
    parser.add_argument('--validate_member')
    parser.add_argument('--taskFiles')
    # parser.add_argument('--w2v')
    # parser.add_argument('--ws_child')
    parser.add_argument('--ws_path')
    # parser.add_argument('--ws_catcode')
    parser.add_argument('--percentages', nargs='*', type=int)
    parser.add_argument('--numOfChild', type=int)
    parser.add_argument('--logPath')

    """
    # usage 4: plot result of validate membership
    $ python nball.py  --plot_validate_member /Users/tdong/data/glove/memberValidation/membershipPredictionResult.txt --numOfChild 10 --percentages 5 10 20 30 40 50 60 70 80 90 
    """
    parser.add_argument('--plot_validate_member')

    """
    # usage 5: consistency analysis 1: deviation of word-stems
    $ python nball.py  --std_stem /Users/tdong/data/glove/wordstem.std --dim 50 --w2v /Users/tdong/data/glove/glove.6B.50d.txt --ballStemFile /Users/tdong/data/glove/glove.6B.50Xball.words --ball /Users/tdong/data/glove/glove.6B.50Xball.V10.txt

    """
    parser.add_argument('--std_stem')
    parser.add_argument('--ballStemFile')
    parser.add_argument('--dim', type=int)

    args = parser.parse_args()
    if args.train_nball and args.w2v and args.ws_child and args.ws_catcode and args.log:
        outputPath, nballFile = os.path.split(args.train_nball)
        logFile = os.path.join(outputPath, 'traing.log')
        outputPath = os.path.join(outputPath, "data_out")

        wsChildrenDic, word2vecDic, wscatCodeDic = initialize_dictionaries(word2vecFile=args.w2v,
                                                                           catDicFile=args.ws_catcode,
                                                                           wsChildrenFile=args.ws_child)

        train_word2ball(root="*root*", outputPath=outputPath, wsChildrenDic=wsChildrenDic,
                        word2vecDic=word2vecDic, wscatCodeDic=wscatCodeDic, logFile=logFile,
                        word2ballDic=dict(),
                        outputBallFile=args.train_nball)

    if args.neighbors and args.ball:
        ballDic = load_ball_embeddings(args.ball)
        if args.num:
            num = args.num
        else:
            num = 10

        wlst = args.neighbors

        nearest_neighbors_of_word_sense(tlst=wlst, dic=ballDic, numOfNeighbors=num)

    if args.validate_member and args.taskFiles and args.w2v and args.ws_child and args.ws_path and args.ws_catcode \
            and args.percentages and args.numOfChild and args.logPath:
        membership_prediction_in_batch(args.percentages, NumOfChild=args.numOfChild, trainingTestingFile=args.taskFiles,
                                       outputFile=args.validate_member, NodeChildrenFile=args.ws_child,
                                       catPathFile=args.ws_path, catFingerPrintFile=args.ws_catcode,
                                       w2vFile=args.w2v, logPath=args.logPath)

    if args.plot_validate_member and args.percentages and args.numOfChild:
        show_membership_prediction_result(filePat=args.plot_validate_member, pers=args.percentages)

    if args.std_stem and args.ballStemFile and args.dim and args.ball and args.w2v:
        maximum_deviation(ofile=args.std_stem, word2vecFile= args.w2v,
                          ballFile=args.ball, ballStemFile=args.ballStemFile, dim=args.dim)

    if args.zero_energy and args.ball and args.ws_child:
        wsChildrenDic = dict()
        with open(args.ws_child, 'r') as chfh:
            for ln in chfh:
                wlst = ln[:-1].split()
                wsChildrenDic[wlst[0]] = wlst[1:]
        testing_whole_family(outputPath=args.zero_energy, wsChildrenDic=wsChildrenDic,
                             word2ballDic=dict(), outputBallFile=args.ball)


if __name__ == "__main__":
    main()