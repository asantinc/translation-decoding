#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import pdb

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1000, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]


def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def printStacks(stacks,f):
  for i,stack in enumerate(stacks):
    if len(stack)>0:  
      print '--------------- Stack: '+ str(i)+'--------------- '
      print ' '.join(f)
      for h in sorted(stack.itervalues(),key=lambda h: -h.logprob):
        sentence = extract_english(h)
        bits = h.bitmap
        print ', '.join(str(x) for x in bits) +' \n '+ sentence +' == '+str(h.logprob)
      
def updateBitmap(bitmap, startA, middle):
  bits = list(bitmap)
  for i in range(startA, middle):
    bits[i] = 1
  return bits

def getMissingRange(bitmap):
  for i, s in enumerate(bitmap):
    if s == 0:
      start = i
      break
  for j, e in enumerate(bitmap[start:], start):
    if e == 1:
      end = j
      return (1, start, end)
  return (0, 0,0)


def hypothesize(stacks, sentence, tm, lm, h, bitmap):
  stack_index = sum(bitmap)
  if sentence in tm:				                   
    for phrase in tm[sentence]:		                
      logprob = h.logprob + phrase.logprob	        
      lm_state = h.lm_state			                
      for word in phrase.english.split():		            
        (lm_state, word_logprob) = lm.score(lm_state, word)	
        logprob += word_logprob			
      logprob += lm.end(lm_state) if stack_index == len(f) else 0.0
      new_hypothesis = h._replace(logprob=logprob, lm_state=lm_state, predecessor=h, phrase=phrase, bitmap=bitmap)

      if lm_state not in stacks[stack_index] or stacks[stack_index][lm_state].logprob < logprob:
        stacks[stack_index][lm_state] = new_hypothesis

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))

for i,f in enumerate(french):
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitmap")
  initial_bitmap = [0] * len(f)
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, initial_bitmap)
  stacks = [{} for _ in f] + [{}]		
  stacks[0][lm.begin()] = initial_hypothesis

  for startA, stack in enumerate(stacks[:-1]):
    printStacks(stacks, f)
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]:
      gap = getMissingRange(h.bitmap)
      if (gap[0]):              #there's a gap that, fill it and create new hypotheses
        bitmap = updateBitmap(h.bitmap, gap[1], gap[2])
        hypothesize(stacks, f[gap[1]:gap[2]], tm, lm, h, bitmap)
      
      else:                     # no gap, split sentence in two
          for end in range(startA+2, len(f)+1):
            for middle in range(startA+1, end+1):
              bitmap = updateBitmap(h.bitmap, startA, middle)
              hypothesize(stacks, f[startA:middle], tm, lm, h, bitmap)

              if middle < end:  #take the second part, if we're not at the last loop
                bitmap = updateBitmap(h.bitmap, middle, end)
                hypothesize(stacks, f[middle:end], tm, lm, h, bitmap)
                '''
                for mid_2 in range(middle+1,len(f)+1):
                  if  f[mid_2:end] in tm:
                    for phrase in tm[f[mid_2:end]]:
                      bitmap = updateBitmap(h.bitmap, mid_2, end)
                      hypothesize(stacks, f[mid_2:end], tm, lm, h, bitmap)
                '''

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))















