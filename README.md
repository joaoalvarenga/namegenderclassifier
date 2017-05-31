# Name Gender Classifier
A gender classifier based on first names.  
This classifier implements a single layer perceptron as main classifier.  
It uses final trigrams and character frequency as features into the classifier.

## Dataset
With brazilian names dataset, my current numbers are:
```
Accuracy: 0.759988
Precision: 0.753677
Recall: 0.756184
F1: 0.754929
```

# Quick start
## Requirements
This project uses Python 3 specifications
Install all project dependencies via pip after cloning project
```
$ python setup.py install
$ pip install -r requirements.txt
```
## Training example
```python
from genderclassifier import GenderClassifier
import pandas as pd

dataset = pd.read_csv("data/nomes.csv").values

classifier = GenderClassifier()
classifier.train(dataset)
classifier.save("models/example")
precision, recall, accuracy, f1 = classifier.evaluate(dataset)
print("Accuracy: %f" % accuracy)
print("Precision: %f" % precision)
print("Recall: %f" % recall)
print("F1: %f" % f1)
```

## Predicting
```python
from genderclassifier import GenderClassifier
classifier = GenderClassifier()
classifier.load("models/example")
name = input()
while name is not "q":
    pred = classifier.predict([name])
    print("%s - %s" % (name, pred))
    name = input()
```

# License
MIT License

# Contributing

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Steps to contribute:

- Make your awesome changes
- Submit pull request
- You can also help sharing better datasets ;)
