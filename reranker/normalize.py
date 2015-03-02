import sys
from collections import defaultdict
import pdb

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return float(ss)

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/float(n) # the population variance
    return pvar**0.5

'''
Takes a file with several groups of num_sents scores. For each group, it finds its mean and std. dev., and then normalizes them
The normalized values are written out to output_fname
'''
def normalize(input_fname, output_fname, num_sents=100):
    f = open(input_fname, 'r')
    lines = f.read().splitlines()
    feature_values = defaultdict()
    counter = 0

    for i, line in enumerate(lines):
        if ((i % num_sents) == 0):
            feature_values[i/num_sents] = []
        feature_values[i/num_sents].append(float(line))

    mean_feats = defaultdict()
    std_feats = defaultdict()

    #find the mean and variance for each sentence
    for i in range(len(lines)/num_sents):
        mean_feats[i] = mean(feature_values[i])
        std_feats[i] = pstdev(feature_values[i])

    #normalize the data for each batch of translations
    f_out = open(output_fname, 'w')
    for i, line in enumerate(lines):
        if (std_feats[i/num_sents] != 0):
            new_val = ((float(line.strip())-mean_feats[i/num_sents])) /std_feats[i/num_sents]
        else: # if there is zero std.dev and we centre all data around zero, the new_value must be zero
            new_val = 0
        f_out.write(str(new_val)+'\n')
    
    f_out.close()
    f.close()
    return [mean_feats, std_feats]

feat_unnorm = 'dev+test/unnorm/'
feat_norm = 'dev+test/norm/'
for var in ['lex', 'lm', 'tm']:
    normalize(feat_unnorm+var+'.out', feat_norm+var+'.out')

feat_unnorm = 'train/unnorm/'
feat_norm = 'train/norm/'
for var in ['lex', 'lm', 'tm', 'ibm', 'diag']:
    normalize(feat_unnorm+var+'.out', feat_norm+var+'.out')




    
