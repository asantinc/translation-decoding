#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import variance


def rerank(mean_feats, std_feats, lex=1, tm=1, lm=1, length=1, outfilename='default', normalize=True):
    outfile = open(outfilename+'.out','w')
    outfile_location = outfilename+'.out'
    hypotheses = open("workdir/dev+test.with-len.txt")

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
          if normalize:
            score += weights[k] * ((float(v)-mean_feats[(s,k)])/float(std_feats[(s,k)]))
          else:
            score += weights[k] * float(v)
        if score > best_score:
          (best_score, best) = (score, hyp)
      try: 
        outfile.write("%s\n" % best)
      except (Exception):
        sys.exit(1)
    return outfile_location



