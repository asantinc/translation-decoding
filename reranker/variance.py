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

def normalize(input_fname, output_fname, num_sents=100):
    f = open(input_fname, 'r')
    feature_values = defaultdict()
    counter = 0
    for i, line in enumerate(f):
        feature_values[i/100].append(float(line.strip()))

    mean_feats = defaultdict()
    std_feats = defaultdict()

    #find the mean and variance for each sentence
    for i in len(f)/100:
        mean_feats[i/100] = mean(feature_values[i/100])
        std_feats[i/100] = pstdev(feature_values[i/100])

    #normalize the data for each batch of translations
    f_out = open(output_fname, 'w')
    for i, line in enumerate(f):
        new_val = (float(line.strip())-mean_feats[i/100])/std_feats[i/100]
        f_out.write(str(new_val)+'\n')
    
    f_out.close()
    f.close()
    return [mean_feats, std_feats]




hypotheses = open("workdir/dev+test.with-len.txt")
all_hyps = [hyp.split(' ||| ') for hyp in hypotheses]

#contains the list of values for each feature
#feature_values = {'p(e)': [], 'p(e|f)': [], 'p_lex(f|e)': [], 'len':[]}
feature_values = {}

all_feats = set()
for hyp in all_hyps:
    _, _, feats = hyp
    for feat in feats.split():
        k,_ = feat.split('=')
        all_feats.add(k)

num_sents = len(all_hyps) / 100
for s in xrange(0, num_sents):
    hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    (best_score, best) = (-1e300, '')
    for (num, hyp, feats) in hyps_for_one_sent:
        score = 0.0
        for feat in feats.split(' '):
            (k, v) = feat.split('=')
            if (s,k) not in feature_values:
                feature_values[(s,k)] = []
            feature_values[(s,k)].append(float(v))

mean_feats = defaultdict()
std_feats = defaultdict()

for s in xrange(0, num_sents):
    mean_feats[(s,'p(e)')] = mean(feature_values[(s,'p(e)')])
    std_feats[(s,'p(e)')] = pstdev(feature_values[(s,'p(e)')])
    mean_feats[(s,'p(e|f)')] = mean(feature_values[(s,'p(e|f)')])
    std_feats[(s,'p(e|f)')] = pstdev(feature_values[(s,'p(e|f)')])
    mean_feats[(s,'p_lex(f|e)')] = mean(feature_values[(s,'p_lex(f|e)')])
    std_feats[(s,'p_lex(f|e)')] = pstdev(feature_values[(s,'p_lex(f|e)')])
    mean_feats[(s,'len')] = mean(feature_values[(s,'len')])
    std_feats[(s,'len')] = pstdev(feature_values[(s,'len')])

#print 'Length:'+str(len_mean)+','+str(len_var)+', TM:'+str(tm_mean)+','+str(tm_var)+', LM:'+str(lm_mean)+','+str(lm_var)+', LxM:'+str(lex_mean)+','+str(lex_var)

    
