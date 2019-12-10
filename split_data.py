import os
from sklearn.model_selection import ShuffleSplit

if not os.path.isdir('train'):
    os.mkdir('train')
if not os.path.isdir('dev'):
    os.mkdir('dev')
if not os.path.isdir('test'):
    os.mkdir('test')

lines = []
for filename in os.listdir('features'):
    f = open('features/'+filename)
    for line in f.readlines():
        lines.append(line)
    f.close()

selector =  ShuffleSplit(train_size=0.9, test_size=0.1)

#split off testing set
for x, y in selector.split(lines):
    train = [lines[i] for i in x]
    test = [lines[i] for i in y]
    break

#split off validation set
for x, y in selector.split(train):
    train = [lines[i] for i in x]
    validate = [lines[i] for i in y]
    break

f = open('train/data.txt', 'w+')
for line in train:
    f.write(line)
f.close()

f = open('dev/data.txt', 'w+')
for line in validate:
    f.write(line)
f.close()

f = open('test/data.txt', 'w+')
for line in test:
    f.write(line)
f.close()
