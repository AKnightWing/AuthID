# AuthID
A Python3 program to identify the author of an unknown text. This is done by analysing the charactertistic ngram frequencies of the authors' works in the training set, and the matched to data in the test set.

## Prerequisites

You need python3 and nltk installed. Further, there should be 2 directories - "Train Data" and "Test Data", present in the directory where the py file is located.

For example, for 3 authors, the directory structure is as follows:
<pre>
<ANY FOLDER>
├── AuthID.py
├── Train Data/
│   ├── Author#1/
│   │     ├──── known_text_1.txt
│   │     ├──── known_text_2.txt
│   │     └──── known_text_3.txt
│   │
│   ├── Author#2/
│   │     ├──── known_text_1.txt
│   │     └──── known_text_2.txt
│   │
│   └── Author#3/
│         ├──── known_text_1.txt
│         ├──── known_text_2.txt
│         ├──── known_text_3.txt
│         └──── known_text_4.txt
│    
└── Test Data/
    ├── unknown_text1.txt
    ├── unknown_text2.txt
    ├── unknown_text3.txt
    └── unknown_text4.txt
</pre>

## Running it from command line
For Linux/Unix:
```python
python3 AuthID.py
```
For Windows:
```
python AuthID.py
```

## Github Link
https://github.com/siddharthchaini/AuthID

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Authors
Siddharth Chaini
&
Siddharth Bachoti

IISER Bhopal
