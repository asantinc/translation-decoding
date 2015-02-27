#!/usr/bin/env python
import optparse
import sys

import bleu
def compute_bleu(file_name):
    f = open(file_name,'r')
    refs = [line.strip().split() for line in open("data/dev.ref")]
    hyps = [line.strip().split() for line in f]

    stats = [0 for i in xrange(10)]
    for (r,h) in zip(refs, hyps):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h,r))]
    return (100 * bleu.bleu(stats))

def individual_bleu(ref, hyp):
    stats =  bleu.bleu_stats(hyp,ref)
    return (100 * bleu.bleu(stats))

