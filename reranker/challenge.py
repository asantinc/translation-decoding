from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import sys


##############################
#       Challenge
##############################

'''
feat_scores = defaultdict()
feat_weights = {'lex': 1, 'tm': 1, 'lm': -1}
outname = 'output/q5/???.out'

for feat in feat_weights.keys():
    if feat not in feat_scores:
        f = open(feat_unnorm+feat, 'r')
        scores = f.read().splitlines()        
        feat_scores[feat] = scores

sent_f = open('data/dev+test.100best', 'r')
sentences = sent_f.read().splitlines()
f = rerank(sentences, feat_scores, feat_weights, outname)
sys.stderr.write('Q1: '+str(compute_bleu(f)) + '\n')
'''
