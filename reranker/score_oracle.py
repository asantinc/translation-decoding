from bleu import *

def build_ordered_list(filename):    
    '''
    Opens a file, produces a list with lines, and closes the file. This helps ensure that files close.
    '''
    f = open(filename,'r')
    ordered = [line.rstrip('\n') for line in f]
    f.close()
    return ordered

def score_oracle(loc):
    # Build up an ordered list of references
    references = build_ordered_list(loc+'ref.out')

    oracle = build_ordered_list(loc+'oracle.out')
    outfile = open(loc+'oracle_bleu.out','w')
    for i in range(len(oracle)):
        bleu_score = individual_bleu(oracle[i], references[i])
        outfile.write(str(bleu_score)+'\n')
    outfile.close()

score_oracle('train/')
score_oracle('dev+test/')
