import nltk
from nltk import tree
import os

def find_demprn(parse, idx):
    demprn = False
    for t in parse.subtrees():
        if t.label() == 'NP' and len(t) == 1:
            if t[0].label() == 'DT' and len(t[0]) == 1:
                if t[0][0] in ['this', 'that', 'This', 'That']:
                    t[0][0] = '*tok{0}* '.format(idx) + t[0][0] + ' *tok{0}*'.format(idx)
                    demprn = True
                    break
    tree_str = str(parse).replace('\n', ' ').replace('  ', '')
    return(tree_str, demprn)

def main():
    #get parses
    for filename in os.listdir('parses'):
        print("processing "+filename)

        f = open('parses/'+filename)
        outf = open('labeled_tok/'+filename, 'w+')
        idx = 0

        #label tokens
        for line in f.readlines():
            try:
                parse = tree.Tree.fromstring(line[:-1])
                tree_str, demprn = find_demprn(parse, idx)
                output = tree_str+'\n'
                idx += demprn
            except:
                #speaker label
                output = line

            outf.write(output)

        f.close()
        outf.close()

main()
