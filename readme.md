# Install the package

* for Ubuntu platform please first install python3-tk
```
sudo apt-get install python3-tk
```

* for Ubuntu or Mac platform type:

```
$ git clone https://github.com/gnodisnait/nball4tree.git
$ cd nball4tree
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

```

# Experiment 1:  Training and evaluating nball embeddings
## Experiment 1.1: Training nball embeddings
* [datasets for training nball embeddings](https://drive.google.com/file/d/1RYMH93gmaQivY1aJdWCVpey1nZ0YL9dl/view?usp=drive_link)
* download glove.6B.50d.txt from the GloVe webpage https://nlp.stanford.edu/projects/glove/ 
* shell command for running the nball construction and training process
```
% you need to create an empty file nball.txt for output

$ python nball.py --train_nball /Users/<user-name>/data/glove/nball.txt --w2v /Users/<user-name>/data/glove/glove.6B.50d.txt  --ws_child /Users/<user-name>/data/glove/wordSenseChildren47634.txt  --ws_catcode /Users/<user-name>/data/glove/glove.6B.catcode.txt  --log log.txt
% --train_nball: output file of nball embeddings
% --w2v: file of pre-trained word embeddings
% --ws_child: file of parent-children relations among word-senses
% --ws_catcode: file of the parent location code of a word-sense in the tree structure
% --log: log file, shall be located in the same directory as the file of nball embeddings
```
The training process can take around 6.5 hours. 


## Experiment 1.2: Checking whether tree structures are perfectly embedded into word-embeddings
* main input is the output directory of nballs created in Experiment 1.1
* shell command for running the nball construction and training process
```
$ python nball.py --zero_energy <output-path> --ball <output-file> --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt
% --zero_energy <output-path> : output path of the nballs of Experiment 1.1, e.g. ```/Users/<user-name>/data/glove/data_out```
% --ball <output-file> : the name of the output nball-embedding file
% --ws_child /Users/<user-name>/data/glove/wordSenseChildren.txt: file of parent-children relations among word-senses
```
The checking process can take around 2 hours.
* result

If zero-energy is achieved, a big nball-embedding file will be created ```<output-path>/<output-file>```
otherwise, failed relations and word-senses will be printed.

** Test result at Mac platform:
![img|630x420](https://github.com/gnodisnait/nball4tree/blob/master/pic/success_result.png)
** Test result at Ubuntu platform:
![](https://github.com/gnodisnait/nball4tree/blob/master/pic/ubuntu_result.png)
 
- [nball embeddings with 47634 balls](https://drive.google.com/file/d/1cOW8kF30VoPW6WxTCbpstZI15z-pz7_O/view?usp=drive_link)

- [nball embeddings with 54310 balls]

# Experiment 2: Observe neighbors of word-sense using nball embeddings
* [pre-trained nball embeddings](https://drive.google.com/file/d/1cOW8kF30VoPW6WxTCbpstZI15z-pz7_O/view?usp=drive_link)
```
$ python nball.py --neighbors beijing.n.01 berlin.n.01  --ball /Users/<user-name>/data/glove/glove.6B.50Xball.V10.txt  --num 6
% --neighbors: list of word-senses
% --ball: file location of the nball embeddings
% --num: number of neighbors
```

* Results of nearest neighbors look like below:

 <a href="url"><img src="https://github.com/gnodisnait/nball4tree/blob/master/pic/nbneighbors.png"   height="700" width="500" ></a></p>

# Experiment 3: Consistency analysis

## deviation of word-stems
* [datasets for analyzing deviation of word-stems]()

* shell command for running the experiment
```
$ python nball.py  --std_stem /Users/<user-name>/data/glove/wordstem.std --dim 50 --w2v /Users/<user-name>/data/glove/glove.6B.50d.txt --ballStemFile /Users/<user-name>/data/glove/glove.6B.50Xball.words --ball /Users/<user-name>/data/glove/glove.6B.50Xball.V10.txt
```
* Result of consistency analysis

<a href="url"><img src="https://github.com/gnodisnait/nball4tree/blob/master/pic/std0.pnd"  height="400" width="500" ></a></p>

# Experiment 4: Validating unknown word-senses or words

* [datasets for validating unknown word-sense or words]()
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

* Precision of validating the category of unknown words

<a href="url"><img src="https://github.com/gnodisnait/nball4tree/blob/master/pic/precision.png"  height="400" width="500" ></a></p>

* Recall of validating the category of unknown words

<a href="url"><img src="https://github.com/gnodisnait/nball4tree/blob/master/pic/recall.png"  height="400" width="500" ></a></p>

# NBalls for other languages

* [This code has been used to generate nball embeddings for Arabic, Chinese, Hindi, and Russian languages. Click me to get the Jupyter scripts](https://github.com/p3ml/ai_language_technology)

# Cite

If you use the code, please cite the following paper:

Tiansi Dong, Chrisitan Bauckhage, Hailong Jin, Juanzi Li, Olaf Cremers, Daniel Speicher, Armin B. Cremers, Joerg Zimmermann (2019). *Imposing Category Trees Onto Word-Embeddings Using A Geometric Construction*. **ICLR-19** The Seventh International Conference on Learning Representations, May 6 – 9, New Orleans, Louisiana, USA.

