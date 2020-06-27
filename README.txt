ReadMe File

The given folder contains one file and two folders
1. weights
2. data
3. MLProjectFinalFile.py

1. weights
-------------------
This folder the contains the weights of the trained model.

2. data
-----------------------
This folder contains the dataset hin_combined.csv


3. MLProjectFinalFile.py
-------------------------
This is the code file that contains the implementation for Machine Translation.
To run this file use the following commands

	python MLProjectFinalFile.py

This will read dataset from data/hin_combined.csv, load the weights from weights folder
and randomly choose 5 test sentences and generate output for those sentences.
The out contains the expected sentence, predicted sentence and predicted probabilities
of expected words in sentence.

  python MLProjectFinalFile.py True

The second parameter 'True' denotes starting the training process.
This will read dataset from data/hin_combined.csv, load the weights from weights folder
and start training the network for 10 epochs with learning rate of 0.01

-------------------------------------------------------------------------------
Python modules required include the following
---------------------------------------------
Module                -> Used for
------                --------------
numpy                 -> for mathematical calculations
pandas                -> for loading and preprocessing the dataset
re                    -> to match regular expressiong in text preprocessing
string                -> to use default set of punctuation marks
sklearn.preprocessing -> to use LabelEncoder and OneHotEncoder to generate OneHot vector for words
sklearn.utils         -> to shuffle the dataset
heapq                 -> to get max probabilities from array
sys                   -> to get input from commandline
-------------------------------------------------------------------------------
