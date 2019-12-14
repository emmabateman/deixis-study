# deixis-study
a computational study of the distribution of the pronouns "this" and "that" in podcast transcripts.

This project was developed on Ubuntu 18.04.3 with Python 3.6.9

## Data
The data comes from 4 episodes of _My Brother, My Brother, and Me_ (episodes 481-484). The transcripts were posted on the Maximum Fun website in PDF format. Raw text was obtained by copy+pasting from the PDFs.

clean\_data.py cleans the data by removing file headers and incorrectly copied characters.

parse.py uses the Stanford NLP parser to produce syntax trees for each sentence.

label\_tokens.py automatically marks demonstrative pronouns.

After running the first three scripts, we labeled the referent for each pronoun by hand.

features.py produces feature vectors for each labeled instance. These feature vectors are stored in the features folder

split\_data.py randomly divides the feature vectors into test, dev, and train sets.

## Analysis

Analysis of the data is performed by running analysis.py.

As part of the analysis, 3 prediction models are trained using scikit.

To run predictions on the test set instead of the dev set, use option -t.

_This_ and _is_ can't be contracted due to phonological properties. This means that "next:is" correlates strongly with _this_ and "next:'s" correlates strongly with _that_. This is a problem if we want to discover patterns based on the semantic differences between _this_ and _that_ rather than their phonological differences.

To eliminate the distinction between 's and is, use option -s.
