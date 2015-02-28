#!/usr/bin/env python
import optparse
import sys
import bleu
import pdb

def compute_bleu(hypo, ref="data/dev.ref"):
    f_ref = open(ref,'r')
    f_hypo = open(hypo,'r')
    ref = [line.strip().split() for line in f_ref]
    hyp = [line.strip().split() for line in f_hypo]
    f_hypo.close()
    f_ref.close()

    stats = [0 for i in xrange(10)]
    for (r,h) in zip(ref, hyp):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h,r))]
    return (100 * bleu.bleu(stats))



