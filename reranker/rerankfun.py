#!/usr/bin/env python
import optparse
import sys
import pdb
#from numpy import *
from collections import defaultdict


'''
Ranks each group of translations in 'translations' using the features provided, with their corresponding weights. 
@translations: the list of lists of translations to be scored, one list of translations for each Russian sentence
@translations type: list

@scores: a list of lists of feature vector tuples representing translations, one list of feature vectors for each Russian sentence
@scores type: list

@weights: list of weights for the features
@weights type: list

@outname: file in which the best translations are written to
@outname type: string
'''
def rerank(translations_list, scores, weights, outname):
  outfile = open(outname, 'w')
  (best_score, best) = (-1e300, '')  
  num_sents = (len(translations_list))

  for rus_ind, translations in enumerate(translations_list):  # for all Russian sentences
    (best_score, best) = (-1e300, '')
    for tr_ind, translation in enumerate(translations):  # loop over this sentence's translations
      score = 0.0
      for w_ind, w in enumerate(weights):
        score += w * scores[rus_ind][tr_ind][w_ind] # add the weighted feature score for this translation
      if score > best_score:
        (best_score, best_translation) = (score, translation)
    try:  
      outfile.write("%s\n" % best_translation)
    except (Exception):
      sys.exit(1)

  outfile.close()




def rerank_basic(lex=1, tm=1, lm=1, length=0, outfilename='default'):
    outfile = open(outfilename,'w')
    hypotheses = open("data/dev+test.with-len.txt")

    weights = {'p(e)': lm, 'p(e|f)': tm, 'p_lex(f|e)': lex, 'len':length}
 
    all_hyps = [hyp.split(' ||| ') for hyp in hypotheses]
 
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
          score += weights[k] * float(v)
        if score > best_score:
          (best_score, best) = (score, hyp)
      try:
        outfile.write("%s\n" % best)
      except (Exception):
        sys.exit(1)
    return outfilename







