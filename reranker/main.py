from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import sys
import os

def print_scores(lm, tm, lex, score, result_file, length=0):
    sys.stderr.write(q+': LM='+str(lm)+' TM='+str(tm)+' Lex='+str(lex)+' Len='+str(length)+' BLEU='+str(score) + '\n')
    if length !=0:
        result_file.write(str(length)+','+str(score)+';\n')
    else:
        result_file.write(str(lm)+','+str(tm)+','+str(lm)+','+str(score)+';\n')

feat_norm = 'data/features/norm/train.'
feat_unnorm = 'data/features/unnorm/train.'
features = ['diag','ibm','lex','tm','diag_rev','ibm_rev','lm','untranslated']

################################
#	            Q1
################################
lex = 1
tm = 1
lm = 1
length = 0
outfilename = 'output/q1/fliplm_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
sys.stderr.write('Q1 LM=-1: '+str(compute_bleu(f)) + '\n')


################################
#	            Q2
################################
result_file = open('output/q2/result', 'w')

# Zero
q = 'Q2'
lex = 0
tm = 1
lm = 1
outfilename = 'output/q2/lexZero_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)


lex = 1
tm = 0
lm = 1
outfilename = 'output/q2/tmZero_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)


lex = 1
tm = 1
lm = 0
outfilename = 'output/q2/lmZero_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)

# Flip
lex = -1
tm = 1
lm = 1
outfilename = 'output/q2/lexFlip_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)


lex = 1
tm = -1
lm = 1
outfilename = 'output/q2/tmFlip_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)

lex = 1
tm = 1
lm = -1
outfilename = 'output/q2/lmFlip_unnorm.out'
f = rerank_basic(lex, tm, lm, length, outfilename)
score = compute_bleu(f)
print_scores(lm, tm, lex, score, result_file)

result_file.close()

################################
#	            Q4
################################

length = 0
q='Q4'
result_file = open('output/q4/result','w')
for lex in range(1, 16, 2):
    for lm in range(1, 16, 2):
        for tm in range(1, 16, 2):
            outfilename = 'output/q4/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
            f = rerank_basic(lex, tm, lm, length, outfilename)
            score = compute_bleu(f)
            os.remove(f)            #we don't want to keep the file
            print_scores(lm, tm, lex, score, result_file)
result_file.close()

################################
#	            Q5
################################
lex = 1
tm = 1
lm = 1

q='Q5'
result_file = open('output/q5/result','w')
for length in range(1, 16, 2):
    outfilename = 'output/q5/'+'len'+str(length)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    os.remove(f)            #we don't want to keep the file
    print_scores(lm, tm, lex, score, result_file, length)
result_file.close()



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

