#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

def rerank(lex=1, tm=1, lm=1, lngth=1 outfilename='default'):
    outfile = open('workdir/wordcount'+outfilename+'.out','w')
    outfile_location = 'workdir/wordcount'+outfilename+'.out'
    hypotheses = open("data/dev+test.100best")

    weights = {'p(e)': lm, 'p(e|f)': tm, 'p_lex(f|e)': lex}

    all_hyps = [hyp.split(' ||| ') for hyp in hypotheses]

    all_feats = set()
    for hyp in all_hyps:
      _, _, feats = hyp
      for feat in feats.split():
        k,_ = feat.split('=')
        all_feats.add(k)
    sys.stderr.write('Weights:')
    for feat in all_feats:
      sys.stderr.write(' %s=%g' % (feat, weights[feat]))
    sys.stderr.write('\n')
        
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
    return outfile_location

