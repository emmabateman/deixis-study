#clean up text from copy+pasting pdfs

import os

def clean(sent):
    sent = sent.replace('―', '"')
    sent = sent.replace('‖', '"')
    sent = sent.replace('‗', '\'')

    return sent

def main():
    for filename in os.listdir('data'):    
        print('cleaning file {0}'.format(filename))

        sents = []
        header = True
        text = ''

        f = open('data/'+filename)
        for line in f.readlines():
            if not header:
                #process line
                if line.split()[0][-1] == ':':
                    sents.append(clean(text[:-1]))
                    text = ''
                text = text + line[:-1] + ' '
            if line.find('TheMcElroy.family') > 0: #url marks end of header
                header = False
        sents.append(clean(text))
        outf = open('data_clean/'+filename, 'w+')
        for sent in sents:
            outf.write(sent+'\n')

main()
