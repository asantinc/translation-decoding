#!/usr/bin/env python
import optparse
import sys
import pdb
import math #used by math.log
from collections import defaultdict

'''
Update the expected counts c(f,e) and c(e)
'''
def updateMatchCount(c_fe, c_e, foreign, english, transProb):
    #Update the match number between an english word and a foreign word
    if foreign in c_fe:
        if english not in c_fe[foreign]:
            c_fe[foreign][english] = 0
    else:
        c_fe[foreign] = dict()
        c_fe[foreign][english] = 0
    c_fe[foreign][english] += transProb
    #Also update the match number between an english word and any other foreign word
    if english not in c_e:
        c_e[english] = 0
    c_e[english] += transProb

'''
Find the alignment parameters that give the best probability of the data
'''
def findAligments(bitext, transProbDict):
    logLikelihood = 0
    alignments = ''
    for sentence in bitext:
        sentenceMap = ''
        foreignIndex = 0
        for foreign in sentence[0]:
            bestAlignIndex = -1
            bestAlignScore = -1
            bestWord = ''
            alignmentIndex = 0
            for english in sentence[1]:
                if (transProbDict[foreign][english] > bestAlignScore):
                    bestAlignIndex = alignmentIndex
                    bestAlignScore = transProbDict[foreign][english]
                    bestWord = english
                alignmentIndex += 1
            logLikelihood += math.log(bestAlignScore)
            sentenceMap += str(foreignIndex) +'-' + str(bestAlignIndex) +' '
            foreignIndex += 1
        alignments += sentenceMap + '\n' 
    return [alignments, logLikelihood]

'''
Learning the t(f|e) parameters of IBM model 1 using EM
'''
'''
    englishCounts = dict()      
    for sentence in bitext:
        for foreign in sentence[0]:
            for english in sentence[1]:
                #calculate the number of f's per e so t(f | e ) actually sums up to one
                if english in englishCounts:
                    englishCounts[english].add(foreign)
                else:
                    englishCounts[english] = set()
                    englishCounts[english].add(foreign)

	#Set t( f|e) to a uniform prob distribution. If a word 'f' is never seen with 'e' never been seen, we assume it's t (f='f' | e='e') = 0
    for sentence in bitext:            
        for foreign in sentence[0]:
            for english in sentence[1]:            
                if foreign in transProbDict:
                    transProbDict[foreign][english] = 1.00 / (len(englishCounts[english])) 
                
                else:
                    transProbDict[foreign] = dict()
                    transProbDict[foreign][english] = 1.00 / (len(englishCounts[english]))          
'''
def IbmModelOne(bitext, iterations):
    transProbDict = dict()       #t(f| e)
    c_fe = dict()                #estimated count of matches between words f and e over all sentences
    c_e = dict()                 #estimated count of the matches between e and any other f-word over all sentences
    # Count the number of distinct german words that each english word meets accross all sentences
    # This is necessary so we can normalize the initial t(f | e) values
    foreign = set()
    for (f,e) in bitext:
        foreign |= set(f)
    for (f,e) in bitext:
		for fw in f:
			for ew in e:
				transProbDict[fw][ew] = 1.00 / len(set(foreign))
          
    #run EM            
    for it in range(iterations):
        for sentence in bitext:
            for foreign in sentence[0]:
                #find the denominator of delta, the expected increase in alignment count
                sumTransProb = 0
                for english in sentence[1]:     
                    sumTransProb += transProbDict[foreign][english]
                #find delta and update the estimated counts: c_ef and c_e
                for english in sentence[1]:
                    delta = transProbDict[foreign][english]*1.00 / sumTransProb
                    updateMatchCount(c_fe, c_e, foreign, english, delta)
        #calculate the updated translation probabilities based on the estimated counts stored in c_ef and c_e
        for foreign, valueDict in transProbDict.items():                               
            for english, prob in valueDict.items():
                transProbDict[foreign][english] = (c_fe[foreign][english]*1.00) / (c_e[english]) 
    return transProbDict



#Command line options
optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--num_iterations", dest="num_iterations", default=10, type="int", help="Number of iterations to run EM")
(opts, _) = optparser.parse_args()
sys.stderr.write("Training with IBM model 1...")

#Get foreign language || english sentences
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]

#Learn translation probabilities using IBM model 1
sys.stderr.write("Learning with IBM")
transProbDict = IbmModelOne(bitext, opts.num_iterations)         #train our model using IBM1
[alignments, logLikelihood] = findAligments(bitext, transProbDict)

likeF = open('likely.out', 'a')
likeString = 'Iterations: '+str(opts.num_iterations)+' '+str(logLikelihood)+'\n' 
likeF.write(str(likeString))
print alignments

            
    

    
                
            
