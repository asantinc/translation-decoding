#!/usr/bin/env python
import optparse
import sys
import operator
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=200, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--max_iters", dest="max_iters", default=10, type="int", help="Number of iterations in EM algorithm")
(opts, _) = optparser.parse_args()

bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]

# initialise the parameters P(f|e)
theta = defaultdict(float)
french_dict = defaultdict(int)
english_dict = defaultdict(int)

french = set()
for (f,e) in bitext:
    french |= set(f)
for (f,e) in bitext:
    for fw in f:
        for ew in e:
            theta[(fw,ew)] = 1.00/len(french)

# Count the number of distinct words in each language
for (f,e) in bitext:
    for fw in f:
	french_dict[fw] += 1
    for ew in e:
	english_dict[ew]+= 1

# Find the 5 most common and least common English words
sorted_french = sorted(french_dict.items(), key=operator.itemgetter(1))
sorted_english = sorted(english_dict.items(), key=operator.itemgetter(1))
bot_5 = sorted_english[:5]
top_5 = sorted_english[-5:]
ind = int(round(len(sorted_english)/2))
mid_5 = sorted_english[ind:(ind+5)]
print 'French words: ' + str(len(sorted_french)) + '\n'

# start the counter
k = 0

while k < opts.max_iters:
    # increment
    k += 1

    # restart the counts
    fe_count = defaultdict(float)
    e_count = defaultdict(float)

    # compute expected counts
    for (n, (f,e)) in enumerate(bitext):

        for fw in f:

            # initialise and compute normalisation constant
            Z = 0
            
            for ew in e:
                th = theta[(fw,ew)]
                Z += th
            for ew in e:
                th = theta[(fw,ew)]
                # compute expected count
                c = th/Z

                # increment the counts by this amount
                fe_count[(fw,ew)] += c
                e_count[ew] += c
        if n % 100 == 0:
            sys.stderr.write(".")

    for (fw,ew) in fe_count.keys():
        # M-step: recalculate the parameter P(f|e)
        theta[(fw,ew)] = math.exp(math.log(fe_count[(fw,ew)]) - math.log(e_count[ew]))


#countryF = open('q3/morph-country.csv', 'w')
countryL = []
for fw in sorted_french:
    th = theta[(fw[0], 'country')]
    countryL.append((fw, th))
    #countryF.write('%.20f, ' % (th))
#countryF.write('\n')
sorted_country = sorted(countryL, key=lambda tup: tup[1], reverse = True)
print sorted_country[:10]

#countriesF = open('q3/morph-countries.csv', 'w')
countriesL = []
for fw in sorted_french:
    th = theta[(fw[0], 'countries')]
    countriesL.append((fw, th))
    #countriesF.write('%.20f, ' % (th))
#countriesF.write('\n')
sorted_countries = sorted(countriesL, key=lambda tup: tup[1], reverse = True)
print sorted_countries[:10]




