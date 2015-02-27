from random import randint
from math import fabs

def bleu_difference(s1, s2, reference, alpha):
    stats_s1 = bleu.bleu_stats(s1, reference)
    stats_s2 = bleu.bleu_stats(s2, reference)

    s1_score = bleu.bleu(stats_s1)
    s2_score = bleu.bleu(stats_s2)

    return fabs(s1_score-s2_score)


def getSamples(best_candidates, ref_sent, tau, alpha, sample_number):
    for nbest in best_candidates:
        sample = []
        for i in range(tau):
            s1 = nbests[randint(0, len(nbests))]
            s2 = nbests[randint(0, len(nbests))]
            if bleu_difference(s1, s2, ref) > alpha:
                if s1.smoothed_bleu > s2.smoothed_bleu:
                    sample.append(s1, s2)
                else:
                    sample.append(s2, s1)
            else:
                continue
    sample.sort(key=lambda t: t[1]-t[2]) #in ascending order
    return sample[-sample_number:] 


def perceptron(samples, thetas):
    for (s1, s2) in samples:
                


for i = 1 to epochs:
    getSamples()
