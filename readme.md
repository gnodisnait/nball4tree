# Install the package

```
$ git clone https://github.com/gnodisnait/nball4tree.git
$ cd nball4tree
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

```

# Experiment 1:  Training and evaluating nball embeddings
## Experiment 1.1: Training nball embeddings
* [datasets for training nball embeddings](https://drive.google.com/file/d/1V2kBNgxDzFBznkd97UuwDW0OtionpP6y/view?usp=sharing)
* shell command for running the nball construction and training process
```
$ python nball.py --train_nball /Users/<user-name>/data/glove/nball.txt --w2v /Users/<user-name>/data/glove/glove.6B.50d.txt  --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt  --ws_catcode /Users/<user-name>/data/glove/glove.6B.catcode.txt  --log log.txt
% --train_nball: output file of nball embeddings
% --w2v: file of pre-trained word embeddings
% --ws_child: file of parent-children relations among word-senses
% --ws_catcode: file of the parent location code of a word-sense in the tree structure
% --log: log file, shall be located in the same directory as the file of nball embeddings
```
The training process can take around 8.5 hours.

## Experiment 1.2: Evaluating nball embeddings
* main input is the output directory of nballs created in Experiment 1.1
* shell command for running the nball construction and training process
```
$ python nball.py --zero_energy <output-path> --ball <output-file> --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt
% --zero_energy <output-path> : output path of the nballs of Experiment 1.1, e.g. ```/Users/<user-name>/data/glove/data_out```
% --ball <output-file> : the name of the output nball-embedding file
% --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt: file of parent-children relations among word-senses
```
* result
If zero-energy is achieved, one big nball-embedding file will be created ```<output-path>/<output-file>```
otherwise, failed relations and word-senses will be printed.

# Experiment 2: Observe neighbors of word-sense using nball embeddings
* [pre-trained nball embeddings](https://drive.google.com/file/d/176FZwSaLB2MwTOWRFsfxWxMmJKQfoFRw/view?usp=sharing)
```
$ python nball.py --neighbors beijing.n.01 berlin.n.01  --ball /Users/<user-name>/data/glove/glove.6B.50Xball.V10.txt  --num 6
% --neighbors: list of word-senses
% --ball: file location of the nball embeddings
% --num: number of neighbors
```

# Experiment 3: Consistency analysis

## deviation of word-stems
* [datasets for analyzing deviation of word-stems](https://drive.google.com/file/d/17H2bDIopjyAYjk61GOle_hvVDvtxKN64/view?usp=sharing)

* shell command for running the experiment
```
$ python nball.py  --std_stem /Users/<user-name>/data/glove/wordstem.std --dim 50 --ballStemFile /Users/<user-name>/data/glove/glove.6B.50Xball.words --ball /Users/<user-name>/data/glove/glove.6B.50Xball.V10.txt
```

# Experiment 4: Validating unknown word-senses or words

* [datasets for validating unknown word-sense or words](https://drive.google.com/file/d/1JN8eXzjTGsQDi079ZQXqhYu__N2pVQ_w/view?usp=sharing)
* shell command for running the experiment
```
$ python nball.py  --validate_member /Users/<user-name>/data/glove/memberValidation/membershipPredictionResult.txt \
                    --numOfChild 10  --percentages 5 10 20 30 40 50 60 70 80 90  \
                    --taskFiles /Users/<user-name>/data/glove/memberValidation/membershipPredictionTask.txt \
                    --w2v /Users/<user-name>/data/glove/glove.6B.50d.txt \
                    --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt  \
                    --ws_path /Users/<user-name>/data/glove/wordSensePath.txt \
                    --ws_catcode /Users/<user-name>/data/glove/glove.6B.catcode.txt \
                    --logPath /Users/<user-name>/data/glove/logMemberValidate
```

* command for viewing the result of validating unknown word-sense or word
```
$ python nball.py  --plot_validate_member /Users/<user-name>/data/glove/memberValidation/membershipPredictionResult.txt      --numOfChild 10       --percentages 5 10 20 30 40 50 60 70 80 90
```