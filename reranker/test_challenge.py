from challenge import PRO
from compute_bleu_function import *
from math import fabs
import os

'''
Test that the PRO class outputs the same value as the default 27.3509457562
'''
def test1():
    pro = PRO(train_location='dev+test/')
    temp_file = 'temp.out'  #will be removed
    outfile = open(temp_file,'w')

    for r, ref in enumerate(pro.references):
        best, best_t = (-1e300, '')
        for t in pro.translations[r]:
            t_vec = pro.data[ref][t].features
            curr_score, translation_vector = pro.score(t_vec)

            if curr_score > best:
                (best, best_t) = (curr_score, t)
        
        outfile.write(best_t+'\n')

    outfile.close() #if you don't close it and try to read from it again... it only reads 398 lines ;)
    try:
        b = compute_bleu(temp_file)
        assert (fabs(b - 27.3509457562) < 1e-9)
        sys.stderr.write('PASSED: Test 1 \n')        
    except (AssertionError):
        sys.stderr.write('FAILED: Test 1 \n')
    os.remove(temp_file)

test1()
