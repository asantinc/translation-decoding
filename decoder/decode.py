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

def printStacks(stacks):
  for i,stack in enumerate(stacks):
    if len(stack)>0:  
      print '--------------- Stack: '+ str(i)+'--------------- '  
      for s in stack.keys():     
        sentence = extract_english(stack[s])
        bits = stack[s].bitmap
        print ', '.join(str(x) for x in bits) +' == '+ sentence
      
def updateBitmap(bitmap, startA, startB):
  bits = list(bitmap)
  for i in range(startA, startB):
    bits[i] = 1
  return bits

def getMissingRange(bitmap):
  start = -1
  end = -1
  for i, s in enumerate(bitmap):
    if s == 0:
      start = i
      break
  for j, e in enumerate(bitmap[start:], start):
    if e == 1:
      end = j
      return (start, end)
  return -1


def createHypothesis(stacks, stack_index, sentence, tm, lm, bitmap, h):
  #pdb.set_trace()
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
        #print 'Add to stack'+str(stack_index)+':'+str(bitmap_a).strip('[]')


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
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]:
        if (getMissingRange(h.bitmap) < 0):
          for end in range(startA+2, len(f)+1):
            for startB in range(startA+1, end+1):
              partA = f[startA:startB]
              bitmap_a = updateBitmap(h.bitmap, startA, startB)
              createHypothesis(stacks, sum(bitmap_a), partA, tm, lm, bitmap_a, h)

              if startB < end:  #we're not at the last loop
                partB = f[startB:end] 
                bitmap_b = updateBitmap(h.bitmap, startB, end)
                createHypothesis(stacks, sum(bitmap_b), partB, tm, lm, bitmap_b, h)

        else:               #there's a gap
          missed_range = getMissingRange(h.bitmap)
          missed_phrase = f[missed_range[0]:missed_range[1]]
          bitmap_updated = updateBitmap(h.bitmap, missed_range[0],missed_range[1])

          if missed_phrase in tm:				                   
            for phrase in tm[missed_phrase]:			               
              logprob = h.logprob + phrase.logprob	       
              lm_state = h.lm_state			                
              for word in phrase.english.split():		            
                (lm_state, word_logprob) = lm.score(lm_state, word)	
                logprob += word_logprob					# add it to the log_prob
              logprob += lm.end(lm_state) if sum(bitmap_updated) == len(f) else 0.0		
              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, bitmap_updated)
              if lm_state not in stacks[sum(bitmap_updated)] or stacks[sum(bitmap_updated)][lm_state].logprob < logprob:
                 stacks[sum(bitmap_updated)][lm_state] = new_hypothesis

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))















