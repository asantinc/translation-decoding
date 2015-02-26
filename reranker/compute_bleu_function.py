#!/usr/bin/env python
import optparse
import sys

import bleu
def compute_bleu(file_name):
    f = open(file_name,'r')
    ref = [line.strip().split() for line in open("data/dev.ref")]
    hyp = [line.strip().split() for line in f]

    stats = [0 for i in xrange(10)]
    for (r,h) in zip(ref, hyp):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h,r))]
    return (100 * bleu.bleu(stats))

