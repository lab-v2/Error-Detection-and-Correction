# Metacognitive-error-correction
## Links
Paper: https://arxiv.org/abs/2308.14250

Dataset can be download [here](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/downloads/) 

## Table of Contents

1. [Introduction](#1-introduction)
2. [Usage](#3-usage)
3. [Bibtex](#4-bibtex)
4. [License](#5-license)
5. [Contact](#6-contact)

## Introduction
This is the repo for paper **Rule-Based Error Detection and Correction to Operationalize Movement Trajectory Classification**

## Usage
LabelMatrix-TimeDays-TrajectrotyMatrix.py   data process step 1

Instance_Creation_sequence1.py  data process step 2

special_LRCN_data.py    generate train, valid, test data

train_predict.py    train and test the model

speed_rule.py   speed rule

rule_out.py     improve the model performance through rule.


If you only want to use the rule_out.py file, just run this command:
```
python rule_out.py LRCN_F1_no_overlap_sequential no_overlap_sequential_10
```


LRCN_F1_no_overlap_sequential is the results(labels) from neural network

no_overlap_sequential_10 is the rule
These two folders have more files than "python rule_out.py" needs, so just use the files that are needed.

## Bibtex

## License

## Contact
Bowen Xi - bowenxi@asu.edu
Paulo Shakarian - pshak02@asu.edu
