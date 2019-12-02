import nltk
from nltk import tree
from nltk.parse import stanford
import os

WINDOW_SIZE_BEFORE = 3
WINDOW_SIZE_AFTER = 2

os.environ['CLASSPATH'] = '/home/emma/stanford_parser/stanford-parser-2011-06-08/';

parser = stanford.StanfordParser (
    model_path = '/home/emma/stanford_parser/stanford-parser-2011-06-08/englishPCFG.ser.gz',
    path_to_models_jar = '/home/emma/stanford_parser/stanford-parser-2011-06-08/stanford-parser.jar'
            )

def parse_sent(sent):
    for p in parser.raw_parse(sent):
        tree = nltk.tree.Tree.fromstring(str(p))
        return(tree)

def main():
    #init sentence tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #get sentences
    for filename in os.listdir('data_clean'):
        print("processing "+filename)

        f = open('data_clean/'+filename)
        outf = open('parses/'+filename, 'w+')
        for line in f.readlines():
            try:
                name, statement = line.split(':', 1)
                outf.write(name+'\n')
            except:
                statement = line
            sents = tokenizer.tokenize(statement[:-1])
            for sent in sents:
                parse = str(parse_sent(sent)).replace('\n', ' ').replace('  ', '')
#                print(parse)
                outf.write(parse+'\n')

        f.close()
        outf.close()

main()
