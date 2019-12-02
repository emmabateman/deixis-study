import nltk
from nltk import tree
import os
import re

#helper function for tree_properties
def get_referent_properties(t):
    label = t.label()
    det = None
    try:
        if t[0].label() == 'DT':
            det = t[0].leaves()[0].lower()
    except:
        det = None
    return label, det

#get referent features
def tree_properties(example):
    try:
        ref_str = re.sub('\*tok[0-9]*\*', '', example[1])
        ref_tree = tree.Tree.fromstring(ref_str)
    except:
        print("unable to parse syntax tree "+example[1])

    for t in ref_tree.subtrees():
         if '*ref' in t.label():
             t.set_label(re.sub('\*ref.*\*', '', t.label()))
             label, det = get_referent_properties(t)

    features = []
    if label:
        features.append('label:'+label)
    if det:
        features.append('det:'+det)
    return features

#get words surrounding token
def leaf_properties(example):
    try:
        tok_tree = tree.Tree.fromstring(example[0])
    except:
        print("unable to parse syntax tree "+example[0])

    #get leaves surrounding token
    l = tok_tree.leaves()
    tok_marker = '*tok' + re.search('\*tok(.+?)\*',example[0]).group(1) + '*'
    tok_idx = l.index(tok_marker)
    l.remove(tok_marker)
    l.remove(tok_marker)
    if tok_idx > 0:
        prev1 = l[tok_idx-1]
    else:
        prev1 = None
    if tok_idx > 1:
        prev2 = l[tok_idx-2]
    else:
        prev2 = None
    try:
        next1 = l[tok_idx+1]
    except:
        next1 = None
    try:
        next2 = l[tok_idx+2]
    except:
        next2 = None

    features = []
    if prev1:
        features.append('prev1:'+prev1)
    if prev2:
        features.append('prev2:'+prev2)
    if next1:
        features.append('next1:'+next1)
    if next2:
        features.append('next2:'+next2)
    return features

#check if token has a referent
def ref(example):
    if example[1] == None:
        return 0
    else:
        return 1

#helper function for get_indv_features
def feats(example):
    feat_vector = []
    has_ref = ref(example)
    feat_vector.append('ref:'+str(has_ref))
    if has_ref:
        feat_vector += tree_properties(example)
    feat_vector += leaf_properties(example)
    return feat_vector

#find features specific to token or referent
def get_indv_features(tok_dict, ref_dict):
    examples = []
    for key in tok_dict.keys():
        if key in ref_dict:
            examples.append((tok_dict[key], ref_dict[key]))
        else:
            examples.append((tok_dict[key], None))

    feat_mat = []
    for example in examples:
        feat_mat.append(feats(example))

    return feat_mat

#propogate asterisks down from marked nodes
def propogate(t, label):
    if not isinstance(t, tree.Tree):
        return t
    if '*' in t.label():
        label = True
    for i in range(len(t)):
        if not isinstance(t[i], tree.Tree):
            if label:
                t[i] = '*'+t[i]+'*'
        else:
            t[i] = propogate(t[i], label)
    return t

#helper function for tok_counts
#argument is a tree
def tok_counts_left(t):
    count = 0
    for token in t.leaves():
        if '*' in token:
            return count
        else:
            count += 1
    return count

#helper function for tok_counts
#argument is a tree
def tok_counts_right(t):
    count = 0
    leaves = t.leaves()
    leaves.reverse()
    for token in leaves:
        if '*' in token:
            return count
        else:
            count += 1
    return count

# count tokens on either side of an annotation
# should be passed a parse that only contains one annotation
# argument is a string or tree
def tok_counts(t):
    try:
        #propogate labels
        t = propogate(tree.Tree.fromstring(t), False)
        return tok_counts_left(t), tok_counts_right(t)
    except:
        try:
            #propogate labels
            t = propogate(t, False)
            return tok_counts_left(t), tok_counts_right(t)
        except:
            return 0, 0

#find distance between token and referent in single sentence
def find_dist(t):
    #find reference and token markers in children
    for i in range(len(t)):
        if '*ref' in str(t[i]):
            ref_idx = i
        if '*tok' in str(t[i]):
            tok_idx = i

    #get distance
    if ref_idx == tok_idx:
        return find_dist(t[ref_idx])
    elif ref_idx < tok_idx:
        dist = tok_counts(t[ref_idx])[1] + tok_counts(t[tok_idx])[0]
        for child in t[ref_idx+1:tok_idx]:
            dist += len(child.leaves())
        return dist
    else:
        dist = tok_counts(t[ref_idx])[0] + tok_counts(t[tok_idx])[1]
        for child in t[tok_idx+1:ref_idx]:
            dist += len(child.leaves())
        return dist

#get the number of tokens in a chunk of parses
def length(parses):
    l = 0
    for parse in parses:
        if parse.find('(') < 0:
            #not a parse
            continue
        t = tree.Tree.fromstring(parse)
        for leaf in t.leaves():
            if not '*' in leaf: #ignore annotations
                l += 1
    return l

#find features related to distance between token and referent
def get_distance_features(tok_dict, ref_dict, parses):
    feat_mat = []

    for key in tok_dict.keys():
        if not key in ref_dict:
            feat_mat.append([])
            continue

        tok_marker = '*tok'+key+'*'
        ref_marker = '*ref'+key+'*'

        same_sent = False
        same_speaker = True
        dist = 0
        cataphora = False

        #check if token and referent are in same sentence
        if tok_dict[key] == ref_dict[key]:
            same_sent = True
            if tok_dict[key].find(tok_marker) < ref_dict[key].find(ref_marker):
                cataphora = True
            dist = find_dist(tree.Tree.fromstring(tok_dict[key]))
        else:
            tok_loc = parses.index(tok_dict[key])
            ref_loc = parses.index(ref_dict[key])
            gap = parses[min(tok_loc,ref_loc)+1:max(tok_loc,ref_loc)]

            #check if token and referent are said by the same person
            speaker1 = speaker2 = ''
            idx1 = tok_loc
            idx2 = ref_loc
            while idx1 > 0:
                if parses[idx1].find('(') < 0: #no parentheses
                    speaker1 = parses[idx1]
                    break
                idx1 -= 1
            while idx2 < len(parses):
                if parses[idx2].find('(') < 0: #no parentheses
                    speaker2 = parses[idx2]
                    break
                idx2 += 1
            if speaker1 != speaker2:
                same_speaker = False

            #find distance
            if tok_loc < ref_loc:
                cataphora = True
                dist = tok_counts(tok_dict[key])[1] + length(gap) + tok_counts(ref_dict[key])[0]
            else:
                cataphora = False
                dist = tok_counts(ref_dict[key])[1] + length(gap) + tok_counts(tok_dict[key])[0]

        features = []
        features.append('same_sent:'+str(int(same_sent)))
        features.append('same_speaker:'+str(int(same_speaker)))
        features.append('dist:'+str(dist))
        features.append('cataphora:'+str(int(cataphora)))
        feat_mat.append(features)

    return feat_mat

#get class labels from dictionary of token sentences
def class_labels(tok_dict):
    labels = []
    for key in tok_dict.keys():
        t = tree.Tree.fromstring(tok_dict[key])
        leaves = t.leaves()
        labels.append(leaves[leaves.index('*tok'+str(key)+'*')+1].lower())
        
    return labels

def main():
    for filename in os.listdir('labeled_ref'):
        print("processing "+filename)
        
        f = open('labeled_ref/'+filename)
        outf = open('features/'+filename, 'w+')

        #find parses with token and referent markers
        tok_dict = {}
        ref_dict = {}
        parses = []
        for line in f.readlines():
            #fix referent labels
            line = re.sub(r'\*ref([0-9]+?)\* ', r'*ref\1*', line)
            line = re.sub(r' \*ref([0-9]+?)\*', r'*ref\1*', line)

            parses.append(line)
            #check for token
            match = re.search('\*tok(.+?)\*', line)
            if match:
                n = match.group(1)
                tok_dict[n] = line
            #check for referent
            match = re.findall('\*ref(.+?)\*', line)
            for m in match:
                ref_dict[m] = line

        #get individual features
        feat_mat = get_indv_features(tok_dict, ref_dict)
        #get comparative features
        dist_mat = get_distance_features(tok_dict, ref_dict, parses)

        #get demonstrative pronouns
        labels = class_labels(tok_dict)

        #write features
        for i in range(len(feat_mat)):
            feat_mat[i] += dist_mat[i]

        for i in range(len(feat_mat)):
            outf.write(labels[i])
            outf.write('\t')
            for feat in feat_mat[i]:
                outf.write(feat + ' ')
            outf.write('\n')

        f.close()
        outf.close()

main()
