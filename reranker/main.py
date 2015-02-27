from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import sys
import os
import pdb

def print_scores(q, lm, tm, lex, score, result_file, length=0):
    sys.stderr.write(q+': LM='+str(lm)+' TM='+str(tm)+' Lex='+str(lex)+' Len='+str(length)+' BLEU='+str(score) + '\n')
    if length !=0:
        result_file.write(str(length)+','+str(score)+';\n')
    else:
        result_file.write(str(lm)+','+str(tm)+','+str(lex)+','+str(score)+';\n')



################################
#	            Q1
################################
def q1(sentences, feat_unnorm, feat_norm):   
    weights = {'lex':1, 'lm': 1, 'tm': 0}
    scores_unnorm = {}
    scores_norm = {}

    for feat in weights.keys():
        f_unnorm_feat = open(feat_unnorm+feat+'.out', 'r')
        f_norm_feat = open(feat_norm+feat+'.out', 'r')
        scores_unnorm[feat] = f_unnorm_feat.read().splitlines()   
        scores_norm[feat] = f_norm_feat.read().splitlines()  
        f_unnorm_feat.close()
        f_norm_feat.close()
               
    rank_unnorm = 'output/q1/fliplm_unnorm.out'
    rank_norm = 'output/q1/fliplm_norm.out'

    rerank(sentences, scores_unnorm, weights, rank_unnorm)
    rerank(sentences, scores_norm, weights, rank_norm)
    score_unnorm = compute_bleu(rank_unnorm)
    score_norm = compute_bleu(rank_norm)
    print score_unnorm
    print score_norm

    result_unnorm = open('output/q1/result_unnorm', 'w')
    result_norm = open('output/q1/result_norm', 'w')
    print_scores('Q1 unnorm:', weights['lm'], weights['tm'], weights['lex'], score_unnorm, result_unnorm)
    print_scores('Q1 norm:  ', weights['lm'], weights['tm'], weights['lex'], score_norm, result_norm)
    result_unnorm.close()
    result_norm.close()


def q1_old():
    result_file = open('output/q1/result', 'w')
    lex = 1
    tm = 0
    lm = 1
    length = 0
    outfilename = 'output/q1/original_unnorm.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores('Q1', lm, tm, lex, score, result_file)

################################
#	            Q2
################################
def q2():
    result_file = open('output/q2/result', 'w')
    # Zero
    q = 'Q2'
    lex = 0
    tm = 1
    lm = 1
    length = 0
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)


    lex = 1
    tm = 0
    lm = 1
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)


    lex = 1
    tm = 1
    lm = 0
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)

    # Flip
    lex = -1
    tm = 1
    lm = 1
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)


    lex = 1
    tm = -1
    lm = 1
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)

    lex = 1
    tm = 1
    lm = -1
    outfilename = 'output/q2/'+'lm'+str(lm)+'tm'+str(tm)+'lex'+str(lex)+'.out'
    f = rerank_basic(lex, tm, lm, length, outfilename)
    score = compute_bleu(f)
    print_scores(q, lm, tm, lex, score, result_file)

    result_file.close()

################################
#	            Q4
################################
def q4():
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
                print_scores(q, lm, tm, lex, score, result_file)
    result_file.close()

################################
#	            Q5
################################
def q5():
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
        print_scores(q, lm, tm, lex, score, result_file, length)
    result_file.close()


sentences = open('dev+test/100best_clean.out', 'r').read().splitlines()
feat_unnorm = 'dev+test/unnorm/'
feat_norm = 'dev+test/norm/'

q1(sentences, feat_unnorm, feat_norm)
q1_old()
q2()
q4()
q5()




