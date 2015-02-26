import sys
from collections import defaultdict

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
    feature_values = defaultdict()
    counter = 0
    for i, line in enumerate(f):
        feature_values[num_sents/100].append(float(line.strip()))

    mean_feats = defaultdict()
    std_feats = defaultdict()

    #find the mean and variance for each sentence
    for i in len(f)/num_sents:
        mean_feats[i] = mean(feature_values[i])
        std_feats[i] = pstdev(feature_values[i])

    #normalize the data for each batch of translations
    f_out = open(output_fname, 'w')
    for i, line in enumerate(f):
        new_val = (float(line.strip())-mean_feats[i/100])/std_feats[i/100]
        f_out.write(str(new_val)+'\n')
    
    f_out.close()
    f.close()
    return [mean_feats, std_feats]

    
