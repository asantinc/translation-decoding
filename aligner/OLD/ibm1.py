#!/usr/bin/env python
import optparse
import sys
import pdb
from collections import defaultdict


#Get the translation probability, or initialize it if it has not been initialized
def getTransProb(transProbDict, english, foreign):
    if foreign in transProbDict:
        if english not in transProbDict[foreign]:
            transProbDict[foreign][english] = 1
    else:
        transProbDict[foreign] = dict()
        transProbDict[foreign][english] = 1
    return transProbDict[foreign][english]


def updateMatchCount(c_ef, c_e, foreign, english, transProb):
    #Update the match number between an english word and a foreign word
    if foreign in c_ef:
        if english not in c_ef[foreign]:
            c_ef[foreign][english] = 0
    else:
        c_ef[foreign] = dict()
        c_ef[foreign][english] = 0
    c_ef[foreign][english] += transProb
    #Also update the match number between an english word and any other foreign word
    if english not in c_e:
        c_e[english] = 0
    c_e[english] += transProb


def IbmModelOne(bitext, ITERATIONS):
    '''
    Learning the t(f|e) parameters of IBM model 1 using EM
    '''
    transProbDict = dict()       #translation probabilities from english to foreign
    c_fe = dict()                #estimated count of matches between words f and e
    c_e = dict()                 #estimated count of the matches between e and any other word

    for it in ITERATIONS:
        for sentence in bitext:
            for foreign in bitext[sentence[0]]:
                #find the denominator of delta
                for english in bitext[sentence[1]]:                                 
                    sumTransProb += getTransProb(transProbDict, english, foreign)
                #find delta and update the estimated counts: c_ef and c_e
                for english in bitext[sentence[1]]:                                
                    transProb = getTransProb(transProbDict, english, foreign)*1.00 / sumTransProb
                    updateMatchCount(c_ef, foreign, english, transProb)
                    updateAllMatchCount(c_e, english, transProb)

        #calculate the updated translation probabilities based on the estimated counts stored in c_ef and c_e
        for foreign in transProbDict:                               
            for english in transProbDict[foreign]:
                transProbDict[foreign][english] = (c_e[foreign][english]*1.00) / (c_e[english]) 
        return transProbDict



#Command line options
optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
#optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
sys.stderr.write("Training with IBM model 1...")

#Get foreign language || english sentences
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]

#Learn translation probabilities using IBM model 1
ITERATIONS = 3
transProbDict = IbmModelOne(bitext, ITERATIONS)         #train our model using IBM1

'''
Use the t(f|e)'s learnt to find best alignment to a target sentence. This is obtained by finding the best alignment for each word in the source sentence independently of the other words.
'''
#TODO: what happens to the alignment of a word to nothing?
for sentence in bitext:
    for foreign in bitext[sentence[0]]:
        bestAlign = -1
        bestAlignScore = -1
        alignmentIndex = 0
        for english in bitext[sentence[1]]:
            align
            if transProbDict[foreign][english] > bestAlignScore:
                
            








    
                
            
