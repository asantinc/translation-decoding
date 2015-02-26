#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict


'''
Ranks each group of sentences in 'sentences' using the features provided, with their corresponding weights. 
@sentences: the list of sentences to be scored
@sentences type: list

@scores: a dictionary containing list of scores per feature
@scores type: dictionary

@weights: dictionary with weight per feature
@weights type: dictionary

@outname: file in which the best sentences are written to
'''
def rerank(sentences, scores, weights, outname):
    outfile = open(outname, 'w')

    (best_score, best) = (-1e300, '')    
    for s, sent in enumerate(sentences):
        #sum score over all features
        score = 0
        for feat, w in weights.iteritems():
          score += w * float(scores[feat][s])
        if score > best_score:
          (best_score, best_sent) = (score, sent)

        if s % 100 == 0 and s > 0:
            try: 
                outfile.write("%s\n" % best_sent)
            except (Exception):
                sys.exit(1)
            (best_score, best_sent) = (-1e300, '')

    outfile.close()
    return outname


def rerank_basic(lex=1, tm=1, lm=1, length=1, outfilename='default'):
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

