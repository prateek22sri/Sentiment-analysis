from nltk.corpus import sentiwordnet as swn
import nltk
import os
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()


directory = "./aclImdb/train/pos"
dir = os.fsencode(directory)

scores = {}
count = {}
for file in os.listdir(dir):
    filename = os.fsdecode(file)
    f = open(directory+'/' + filename,'r')
    text = f.read()

    tagged = nltk.pos_tag(text.split())

    for i in tagged:
        x,y = i
        if 'JJ' in y or 'RB' in y:
            r = x.lower()
            r = lemma.lemmatize(r)
            if r not in scores.keys():
                scores[r] = 1
                count[r] = 1
            else:
                scores[r] += 1
                count[r] += 1


directory = "./aclImdb/train/neg"
dir = os.fsencode(directory)

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    f = open(directory+'/' + filename,'r')
    text = f.read()

    tagged = nltk.pos_tag(text.split())

    for i in tagged:
        x,y = i
        if 'JJ' in y or 'RB' in y:
            r = x.lower()
            r = lemma.lemmatize(r)
            if r not in scores.keys():
                scores[r] = -1
                count[r] = 1
            else:
                scores[r] += -1
                count[r] += 1

countNeg = 0
countPos = 0
for x,y in scores.items():
    if y < 0:
        countNeg += y
    else:
        countPos +=y

for x, y in scores.items():
    if y < 0:
        scores[x] = -1 * (y / countNeg)
    else:
        scores[x] = y / countPos


directory = "./aclImdb/test/neg"
dir = os.fsencode(directory)

total = 0
right= 0
for file in os.listdir(dir):
    localscore = 0
    total +=1
    filename = os.fsdecode(file)
    f = open(directory+'/' + filename,'r')
    text = f.read()

    text = text.split()

    for i in text:
        r = i.lower()
        r = lemma.lemmatize(r)

        if r in scores.keys():
            localscore += scores[r]

    if localscore < 0:
        right +=1

print("Negative:")
print(right/total)


directory = "./aclImdb/test/pos"
dir = os.fsencode(directory)

total = 0
right= 0
for file in os.listdir(dir):
    localscore = 0
    total +=1
    filename = os.fsdecode(file)
    f = open(directory+'/' + filename,'r')
    text = f.read()

    text = text.split()

    for i in text:
        r = i.lower()
        r = lemma.lemmatize(r)

        if r in scores.keys():
            localscore += scores[r]

    if localscore > 0:
        right +=1

print("Positive:")
print(right/total)


