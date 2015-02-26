from rerankfun import *
from compute_bleu_function import *
import numpy as np
weight_range = np.arange(-10,11,0.2)

def countwords(file_name):
    f = open(file_name,'r')
    return sum([len(line.split()) for line in f])

wordcounts = []
for lm_w in weight_range:
    f = rerank(1,1,lm_w,'lm'+str(lm_w))
    compute_bleu(f)
    wordcounts.append(countwords(f))
    
print wordcounts
