from challenge import PRO
from compute_bleu_function import *
from math import fabs
import os


def assert_helper(assert_value, train_location, file_to_score):
    ref_location = train_location+'ref.out'
    try:
        b = compute_bleu(file_to_score, ref=ref_location)
        assert (fabs(b - assert_value) < 1e-9)
        sys.stderr.write('PASSED test 1: '+str(b)+' using '+train_location+' \n')        
    except (AssertionError):
        sys.stderr.write('FAILED test 1 : '+str(b)+' using '+train_location+' \n')
        os.remove(file_to_score)



def test1(train_location='dev+test/', write_bleu=False):
    '''
    Test that the PRO class outputs the same value as the default 27.3509457562
    # TODO: we could add a couple more number tests
    '''
    pro = PRO(train_location, write_bleu=write_bleu)
    file_to_score = 'temp.out'  #will be removed
    outfile = open(file_to_score,'w')

    for r, ref in enumerate(pro.references):
        best, best_t = (-1e300, '')
        for t in pro.translations[r]:
            t_vec = pro.data[ref][t].features
            curr_score, translation_vector = pro.score(t_vec)

            if curr_score > best:
                (best, best_t) = (curr_score, t)
        
        outfile.write(best_t+'\n')
    outfile.close() #if you don't close it and try to read from it again... it only reads 398 lines ;)

    assert_value = 27.3509457562 if train_location == 'dev+test/' else 25.760115946 #expected values found using Adam Lopez's 'rerank' code
    assert_helper(assert_value, train_location, file_to_score)
    

def test2(train_location='dev+test/', write_bleu=False):
    '''
    Test Pro's ranking and BLEU scoring method
    '''
    pro = PRO(train_location=train_location, write_bleu=write_bleu)
    out, b = pro.rank_and_bleu()
    os.remove(out)

    assert_value = 27.3509457562 if train_location == 'dev+test/' else 25.760115946 #expected values found using Adam Lopez's 'rerank' code
    try:
        assert (fabs(b - assert_value) < 1e-9)
        sys.stderr.write('PASSED Test 2: '+str(b)+' using '+train_location+' \n')        
    except (AssertionError):
        sys.stderr.write('FAILED Test 2: '+str(b)+' using '+train_location+' \n')


test1(write_bleu=True)                          #rewrites new bleu scores for dev+test/
test2()                                         # will reuse the bleu scores written to file by test1

test1(train_location='train/', write_bleu=True) #rewrites new bleu scores for train/
test2(train_location='train/')


