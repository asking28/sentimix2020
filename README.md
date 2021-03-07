# Sentimix2020
This repository contains code for our paper [**Voice@{SRIB} at {S}em{E}val-2020 Tasks 9 and 12: Stacked Ensemblingmethod for Sentiment and Offensiveness detection in Social Media**](https://www.aclweb.org/anthology/2020.semeval-1.180) published in SemEval 2020(coaffiliated with CoLING 2020) Sentimix task 9 and 12.
## Abstract 
In social-media platforms such as Twitter, Facebook, and Reddit, people prefer to use code-mixed
language such as Spanish-English, Hindi-English to express their opinions. In this paper, we
describe different models we used, using the external dataset to train embeddings, ensembling
methods for Sentimix, and OffensEval tasks. The use of pre-trained embeddings usually helps in
multiple tasks such as sentence classification, and machine translation. In this experiment, we have
used our trained code-mixed embeddings and twitter pre-trained embeddings to SemEval tasks.
We evaluate our models on macro F1-score, precision, accuracy, and recall on the datasets. We
intend to show that hyper-parameter tuning and data pre-processing steps help a lot in improving
the scores. In our experiments, we are able to achieve 0.886 F1-Macro on OffenEval Greek
language subtask post-evaluation, whereas the highest is 0.852 during the Evaluation Period.
We stood third in Spanglish competition with our best F1-score of 0.756. Codalab username is
asking28.

## Authors
1. Abhishek Singh
2. Surya Pratap Singh Parmar

## Set up
### External libraries
After installing Sklearn, Numpy, Pandas, Tensorflow, Keras, Beautiful Soup, Run below commands to finish setup:
```
pip install focal-loss 
pip install keras-tcn==2.8.3 
pip install keras-multi-head 
pip install keras_metrics 
pip install tqdm 
pip install keras-self-attention
```

### File Structure
This repository contains two folders Sentimix and OffensEval. Sentimix folder contains Jupyter and corresponding Python files of Spanglish and Hinglish. OffensEval Folder contains code files of OffensEval English Task 1,2,3 and Turkish, Arabic, Danish and Greek languages.
### Citation
Please cite us using this bibtex

```
@inproceedings{singh-singh-parmar-2020-voice,
    title = "Voice@{SRIB} at {S}em{E}val-2020 Tasks 9 and 12: Stacked Ensemblingmethod for Sentiment and Offensiveness detection in Social Media",
    author = "Singh, Abhishek  and
      Singh Parmar, Surya Pratap",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.180",
    pages = "1331--1341",
    abstract = "In social-media platforms such as Twitter, Facebook, and Reddit, people prefer to use code-mixed language such as Spanish-English, Hindi-English to express their opinions. In this paper, we describe different models we used, using the external dataset to train embeddings, ensembling methods for Sentimix, and OffensEval tasks. The use of pre-trained embeddings usually helps in multiple tasks such as sentence classification, and machine translation. In this experiment, we have used our trained code-mixed embeddings and twitter pre-trained embeddings to SemEval tasks. We evaluate our models on macro F1-score, precision, accuracy, and recall on the datasets. We intend to show that hyper-parameter tuning and data pre-processing steps help a lot in improving the scores. In our experiments, we are able to achieve 0.886 F1-Macro on OffenEval Greek language subtask post-evaluation, whereas the highest is 0.852 during the Evaluation Period. We stood third in Spanglish competition with our best F1-score of 0.756. Codalab username is asking28.",
}
```

### References
1. https://github.com/SilentFlame/Named-Entity-Recognition/blob/master/Twitterdata/processedTweets.csv <br />
2. https://arxiv.org/pdf/1805.11869.pdf <br />
3. http://ceur-ws.org/Vol-2111/paper5.pdf <br />
4. https://github.com/sahilswami96/SarcasmDetection_CodeMixed/blob/master/Dataset/Sarcasm_tweets.txt <br />
5. https://github.com/sahilswami96/SarcasmDetection_CodeMixed/blob/master/Classification_system/build_feature_vector.py <br />
6. https://www.aclweb.org/anthology/C18-1247.pdf how emotional are you <br />
7. https://arxiv.org/pdf/1905.12516.pdf multiple datasets <br />
