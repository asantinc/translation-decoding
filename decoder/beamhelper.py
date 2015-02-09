import models
from collections import namedtuple, defaultdict
import pdb
import sys

class BeamHelper:
    """A class that has helper functions used in decoding"""

    def __init__(self, lm_file, tm_file, translations, verbosity=0):
        self.tm = models.TM(tm_file, translations)
        self.lm = models.LM(lm_file)
        self.verbosity = verbosity
        self.baseline = False
    
    def set_corpus(self, input, num_sents):
        self.corpus = [tuple(line.strip().split()) for line in open(input).readlines()[:num_sents]]
        self.num_sents = num_sents
        # tm should translate unknown words as-is with probability 1
        for word in set(sum(self.corpus,())):
          if (word,) not in self.tm:
            self.tm[(word,)] = [models.phrase(word, 0.0)]
        return self.corpus

    '''
    Returns a hypothesis. With no parameters, the method returns an initial hypothesis.
    '''
    def get_hypothesis(self, f, logprob=0, total_cost=0, lm_state=0, pred=0, bitmap=0, phrase=0):
        hypothesis = namedtuple("hypothesis", "logprob, total_cost, lm_state, predecessor, bitmap, phrase")
        if f == -1:
            return hypothesis(logprob, total_cost, lm_state, pred, bitmap, phrase)
        else:
            return hypothesis(0.0, 0.0, self.lm.begin(), None, [0 for _ in f], None)

    '''
    Initializes a list of stacks with an initial hypothesis. Return the stacks.
    The stack number matches the length of the sentence parameter. 
    '''
    def init_stacks(self, sentence):
        stacks = [{} for _ in sentence] + [{}] # stacks is a list of dictionaries, one for each french word.
        initial_hypothesis = self.get_hypothesis(sentence) 
        stacks[0][(self.lm.begin(),''.join(['0' for _ in sentence]))] = initial_hypothesis 
        return (stacks, initial_hypothesis)
        
        
    '''
    Builds a cost table for a sentence using baseline decoding.
    Returns the cost table. 
    '''
    def get_future_cost(self, f):
        self.baseline = True
        cost_table = defaultdict()
        lm_state = ()
        for length in range(1, len(f)+1):
            for start in range(0, len(f)-length+1):
                end = length+start
                cost_table[(start, end)] = self.__getBaseLineDecoding(f[start:end]) 
        self.baseline = False
        return cost_table
    
    '''
    Returns the baseline decoding cost of a block
    '''
    def __getBaseLineDecoding(self, block):
        cost_stacks, initial_hypothesis = self.init_stacks(block)
        for i, stack in enumerate(cost_stacks[:-1]):
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:self.num_sents]: # prune
                self.baseline_generator(h,cost_stacks, block)
        winner = max(cost_stacks[-1].itervalues(), key=lambda h: h.logprob)
        #print extract_english(winner)
        return winner.logprob

    '''
    Determines the baseline translation (no permutations) for a sentence 
    '''
    def baseline_generator(self, hyp,stacks, f):
        bmp = hyp.bitmap
        gaps = self.find_gaps(bmp) # gaps is a list of the free contiguous gaps
        self.maybe_write(str(gaps)+'\n',4)
        for gap in gaps:
            for i in xrange(gap[0],gap[0]+1):
                for j in xrange(i+1,gap[1]+1): 
                    if f[i:j] in self.tm:
                        for phrase in self.tm[f[i:j]]:
                            h = self.add_hyp(hyp,phrase,i,j,stacks, f)
                            translation = self.extract_english(h)
                            self.maybe_write(translation + '\n',3)
                            

    '''
    Beam Decoder with permutations: given a hypothesis, it generates hypotheses for any gaps in the transaltion. It permutes       the phrases corresponding to all gaps, thus ensuring that all possible permutations are possible (not only adjacent ones)
    '''
    def generate_hypotheses(self, hyp, cost_table, f, stacks):
        self.cost_table = cost_table
        bmp = hyp.bitmap
        gaps = self.find_gaps(bmp) # gaps is a list of the free contiguous gaps
        self.maybe_write(str(gaps)+'\n',4)
        for gap in gaps:
            for i in xrange(gap[0],gap[1]):
                for j in xrange(i+1,gap[1]+1): 
                    if f[i:j] in self.tm:
                        for phrase in self.tm[f[i:j]]:
                            h = self.add_hyp(hyp,phrase,i,j,stacks, f)
                            translation = self.extract_english(h)
                            self.maybe_write(translation + '\n',3)

    '''
    Add a hypothesis to the stack at the correct stack index location
    '''
    def add_hyp(self, hyp,phrase,i,j,stacks, f):
            logprob = hyp.logprob + phrase.logprob 
            lm_state = hyp.lm_state 
            for word in phrase.english.split(): 
                (lm_state, word_logprob) = self.lm.score(lm_state, word) 
                logprob += word_logprob 
            logprob += self.lm.end(lm_state) if j == len(f) else 0.0
            
            added_bits = []
            for k in range(len(f)):
                if i <= k < j:
                    added_bits.append(1)
                else:
                    added_bits.append(0)
                    
            bitmap = [min(x+y,1) for x, y in zip(hyp.bitmap,added_bits)]
            place = sum(bitmap)
            bitmapstr = ''.join([str(x) for x in bitmap])

            if self.baseline != True:
                total_cost = self.future_cost(bitmap) + logprob
            else:
                total_cost = logprob

            # create a new hypothesis for this phrase, now we have its logprob
            new_hypothesis = self.get_hypothesis(-1, logprob, total_cost, lm_state, hyp, bitmap, phrase) 
            #if lm_state not in stacks[place] or stacks[place][lm_state].logprob < logprob: # second case is recombination
        
            if (lm_state,bitmapstr) not in stacks[place] or stacks[place][(lm_state,bitmapstr)].total_cost < total_cost: # second case is recombination
                stacks[place][(lm_state,bitmapstr)] = new_hypothesis # add the hypothesis to the stack at level j
            return new_hypothesis
            
    '''
    Return the future cost of any gaps in the bitmap
    '''        
    def future_cost(self, bitmap):
        gaps = self.find_gaps(bitmap)  
        cost = 0
        for gap in gaps:
            cost += self.cost_table[tuple(gap)]
        return cost

    '''
    Finds the indeces of stretches of zeros in a bitmap.
    Returns a list of tuples with the (start, end) indeces for each stretch
    '''
    def find_gaps(self, bmp):
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

    def prune(self, stack, p):
        return sorted(stack.itervalues(),key=lambda h: -h.logprob)[:p]

    '''
    Utility functions to extract information about a hypothesis
    '''
    def extract_english(self, h): 
        return "" if h.predecessor is None else "%s%s " % (self.extract_english(h.predecessor), h.phrase.english)
    
    def extract_bitmap(self, h):
        return "" if h.predecessor is None else (str(self.extract_bitmap(h.predecessor)) + '\n' + str(h.bitmap))
        

    def extract_tm_logprob(self, h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + self.extract_tm_logprob(h.predecessor)
        tm_logprob = self.extract_tm_logprob(winner)
        beam.maybe_write("LM = %f, TM = %f, Total = %f\n" % 
        (winner.logprob - tm_logprob, tm_logprob, winner.logprob),1)
        
    '''
    Utility function that determines whether to print out information depending on verbosity level desired. 
    '''
    def maybe_write(self, s,level):
      if self.verbosity >= level:
        sys.stderr.write(s)
        sys.stderr.flush()
        
        