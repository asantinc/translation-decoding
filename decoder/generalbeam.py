#!/usr/bin/env python
import optparse
import sys
import models
import test as t
from collections import namedtuple, defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=10, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbosity",type = "int", default=0,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)

# french is a list of french sentences. each french sentence is represented as a tuple of words
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

def maybe_write(s,level):
  if opts.verbosity >= level:
    sys.stdout.write(s)
    sys.stdout.flush()

def find_gaps(bmp):
    if 0 == sum(bmp):
        return [[0,len(bmp)]]
    gaps = []
    filled_places = [i for i, bit in enumerate(bmp) if bit == 1 ]
    if filled_places[0] > 0:
        gaps.append([0,filled_places[0]])
    for k in range(len(filled_places)-1):
        i = filled_places[k]
        j = filled_places[k+1]
        if i+1 < j:
            gaps.append([i+1,j])
    i = filled_places[-1]
    j = len(bmp)
    if i+1 < j:
        gaps.append([i+1,j])
    return gaps

def prune(stack):
    stacks = sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]


sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that the decoder can consider swapping adjacent phrases.
    hypothesis = namedtuple("hypothesis", "logprob, total_cost, lm_state, predecessor, bitmap, phrase")
    initial_hypothesis = hypothesis(0.0, 0.0, lm.begin(), None, [0 for _ in f], None)
    #print initial_hypothesis.bitmap

    stacks = [{} for _ in f] + [{}] # stacks is a list of dictionaries, one for each french word. 
    stacks[0][(lm.begin(),''.join([str(0) for _ in f]))] = initial_hypothesis # lm.begin() = <s>, the start word
    for_real = False

    def extract_english(h): 
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    

    def generate_hypotheses(hyp):
        bmp = hyp.bitmap
        gaps = find_gaps(bmp) # gaps is a list of the free contiguous gaps
        maybe_write(str(gaps)+'\n',4)
        for gap in gaps:
            for i in xrange(gap[0],gap[1]):
                for j in xrange(i+1,gap[1]+1): 
                    if f[i:j] in tm:
                        for phrase in tm[f[i:j]]:
                            h = add_hyp(hyp,phrase,i,j,stacks)
                            translation = extract_english(h)
                            maybe_write(translation + '\n',3)

    def baseline_generator(hyp,stacks):
        bmp = hyp.bitmap
        gaps = find_gaps(bmp) # gaps is a list of the free contiguous gaps
        maybe_write(str(gaps)+'\n',4)
        for gap in gaps:
            for i in xrange(gap[0],gap[0]+1):
                for j in xrange(i+1,gap[1]+1): 
                    if f[i:j] in tm:
                        for phrase in tm[f[i:j]]:
                            h = add_hyp(hyp,phrase,i,j,stacks)
                            translation = extract_english(h)
                            maybe_write(translation + '\n',3)

    def add_hyp(hyp,phrase,i,j,stacks):
            logprob = hyp.logprob + phrase.logprob # add the logprob of this phrase to the logprob of the sequence up this phrase
            lm_state = hyp.lm_state # language model state is the previous one
            for word in phrase.english.split(): # for each english word in the phrase tuple
                (lm_state, word_logprob) = lm.score(lm_state, word) # find the logprob of moving from previous english state to this english word
                logprob += word_logprob # add the logprob to running total for this phrase
            logprob += lm.end(lm_state) if j == len(f) else 0.0 # if at the end of the sentence, add logprob(english word ends the sentence

            added_bits = []
            for k in range(len(f)):
                if i <= k < j:
                    added_bits.append(1)
                else:
                    added_bits.append(0)
            bitmap = [min(x+y,1) for x, y in zip(hyp.bitmap,added_bits)]
            place = sum(bitmap)
            bitmapstr = ''.join([str(x) for x in bitmap])

            if for_real == True:
                total_cost = future_cost(bitmap) + logprob
            else:
                total_cost = logprob

            # create a new hypothesis for this phrase, now we have its logprob
            new_hypothesis = hypothesis(logprob, total_cost, lm_state, hyp, bitmap, phrase) 
            #if lm_state not in stacks[place] or stacks[place][lm_state].logprob < logprob: # second case is recombination
            
            if (lm_state,bitmapstr) not in stacks[place] or stacks[place][(lm_state,bitmapstr)].total_cost < total_cost: # second case is recombination
                stacks[place][(lm_state,bitmapstr)] = new_hypothesis # add the hypothesis to the stack at level j
            return new_hypothesis

    def getBaseLineDecoding(block):
        initial_hypothesis = hypothesis(0.0, 0.0, lm.begin(), None, [0 for _ in block], None)
        cost_stacks = [{} for _ in block] + [{}]
        cost_stacks[0][lm.begin()] = initial_hypothesis
        for i, stack in enumerate(cost_stacks[:-1]):
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
                baseline_generator(h,cost_stacks)
        winner = max(cost_stacks[-1].itervalues(), key=lambda h: h.logprob)
        #print extract_english(winner)
        return winner.logprob
    
    def futureCostTable():
        cost_table = defaultdict()
        lm_state = ()
        for length in range(1, len(f)+1):
            for start in range(0, len(f)-length+1):
                end = length+start
                cost_table[(start, end)] = getBaseLineDecoding(f[start:end]) 
        return cost_table

    cost_table = futureCostTable()

    def future_cost(bitmap):
        gaps = find_gaps(bitmap)  
        cost = 0
        for gap in gaps:
            cost += cost_table[tuple(gap)]
        return cost

    for_real = True

    for i, stack in enumerate(stacks[:-1]):
        maybe_write('stack ' + str(i)+' contains '+str(len(stack))+' hypotheses'+'\n',2)
        
        for hyp in sorted(stack.itervalues(),key=lambda h: -h.total_cost)[:opts.s]:
            maybe_write(str(hyp.bitmap)+'\n',3)
            generate_hypotheses(hyp)
            
    total_hypotheses = sum([len(stack) for stack in stacks])
    maybe_write('There were ' + str(total_hypotheses) +' hypotheses in total'+'\n',2)

    # the winner is the state in the final stack with highest logprob
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    print extract_english(winner)
    
    
    def extract_bitmap(h):
        return "" if h.predecessor is None else (str(extract_bitmap(h.predecessor)) + '\n' + str(h.bitmap))
    maybe_write(extract_bitmap(winner)+'\n',3)

    def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        maybe_write("LM = %f, TM = %f, Total = %f\n" % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob),1)

