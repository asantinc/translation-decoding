# -*- coding: utf-8 -*-
import pdb

def isEnglish(s):
    try:
        s.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def untranslated_feat(infile, outfile):
    f_out = open(outfile,'w')
    f = (open(infile, 'r'))
    list_sents = [line.split() for line in f]
    #pdb.set_trace()

    for i,s in enumerate(list_sents):
        untrans = []
        untrans = [w for w in s if isEnglish(w)==False]
        f_out.write( str(len(untrans)) + '\n')
    f_out.close()

untranslated_feat( 'data/train.100best_stripped', 'scores.del')
