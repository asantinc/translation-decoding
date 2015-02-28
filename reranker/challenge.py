from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
from bleu import *
import os
import sys
from namedlist import namedlist
import pdb
from math import fabs
from random import randint


##############################
#       Challenge
##############################

class PRO(object):

    def __init__(self,
                  train_location='train/', 
                  tps=100,
                  tau=5000,
                  alpha=0.1,
                  sample_number=100,
                  eta=0.1,
                  epochs=5,
                  weights=None, write_bleu=False ):
        
        self.train_location = train_location
        
        self.tps = tps
        if train_location == 'train/':
            self.num_train = 400
        else:
            self.num_train = 800
            

        #percentron training variables
        self.tau = tau
        self.alpha = alpha
        self.sample_number = sample_number
        self.eta = eta
        self.epochs = epochs
        self.pseudocount = 0.0

        #TODO: do I need this references = None??
        self.references = None

        if weights is None: weights = { 'lm':1. , 'tm':1., 'lex': 1.}  
        self.weights = weights
        self.feats = self.weights.keys()

	    # Build up an ordered list of references
        self.references = self.build_ordered_list(self.train_location+'ref.out')
        
        #Build data array, that contains the translations per reference, with their scores and features
        self.data = self.build_data()
        self.build_features(self.num_train, norm=False)
        self.collect_bleu_scores(write_bleu)   


    #TODO: What order do the weights appear in?
    def set_weights(self, weight_vector):
        '''
        Update the class weights values to a different set of weights, if their lengths match
        '''
        assert len(weight_vector) == len(self.weights)
        self.weights = weight_vector


    def build_ordered_list(self, filename):    
        '''
        Opens a file, produces a list with lines, and closes the file. This helps ensure that files close.
        '''
        f = open(filename,'r')
        ordered = [line.rstrip('\n') for line in f]
        f.close()
        return ordered
        


    def build_data(self, ):
        '''
        Builds a dictionary of dictionaries. The 1st is indexed with a russian sentence. The 2nd with each candidate translation for that reference. 
        The value of the second dictionary is a data_point namedlist, which contains the blue score and the feature_vector values for the translation.
        '''
        self.Data_point = namedlist('Data_point', 'bleu features')
        data = defaultdict()
        #source_trn = open(self.train_location+'src.out','r')

        unstructured_translations = self.build_ordered_list(self.train_location + '100best_clean.out')
        self.translations = self.structure(unstructured_translations)

        for r, ref in enumerate(self.references):
            data[ref] = defaultdict()
            for trans in self.translations[r]:
                data[ref][trans] = self.Data_point(None, None)
        return data  
            

    def build_features(self, num_data, norm=True):
        '''
        Given a dataset, build feature vectors for each of its translations.
        Requirements: a directory with the name dataset, which contains folders called 'norm' and 'unnorm'    
        
        @data_location either 'train' or 'dev+test' (the name of any directory containing data in the correct format)
        @num_data the number of data in the dataset
        @norm True if we want to use the normalised data, False otherwise

        @return a list of lists of feature vectors. Each Russian sentence has a list of feature vectors, one for each 
        of its translations.
        '''
        normalised = 'norm/' if norm else 'unnorm/'
        feature_locations = {loc: (self.train_location + normalised + loc) for loc in os.listdir(self.train_location+normalised) if loc[:-4] in self.feats}
        features = {}
        for k,v in feature_locations.iteritems():
            feature = self.build_ordered_list(v)

            assert len(feature) == num_data * self.tps
            feature = self.structure(feature)
            features[k[:-4]] = feature
        
        for i,ref in enumerate(self.references):
            for j, t in enumerate(self.translations[i]):
                feat_vec = {}
                for f in self.feats:
                    curr_feat = float(features[f][i][j])
                    feat_vec[f] = curr_feat
                self.data[ref][t].features = feat_vec

    def structure(self, dataset):
        '''
        Convert the unstructured datasets which aren't grouped by Russian sentence to a structured list, indexed by Russian sentence.
        '''
        unstructured = dataset
        structured = [[unstructured[j] for j in range(i*self.tps, (i+1)*self.tps)] for i in range(len(unstructured)/self.tps)]
        return structured


    def collect_bleu_scores(self,write=False):
        '''
        Compute smoothed BLEU scores for each translation in the list of translations
        @translations_list a structured list of lists of translations, one list for each Russian sentence
        @return the BLEU scores in the same format, i.e. a list of lists
        '''
        if write:
            outfile = open(self.train_location+'bleu_scores','w')
            for ref in self.references:
                translations = self.data[ref]
                bleu_score_list = []
                for t in translations:
                    bleu_score = individual_bleu(t, ref)
                    translations[t].bleu = bleu_score
                    bleu_score_list.append(bleu_score)
                outfile.write(str(bleu_score_list)+'\n')
            outfile.close()
        else:
            bleu_scores_list = self.build_ordered_list(self.train_location+'bleu_scores')
            for i, ref in enumerate(self.references):
                translations = self.data[ref]
                for j, t in enumerate(translations):
                    bleu_score = bleu_scores_list[i][j]
                    translations[t].bleu = bleu_score
        
          
    def get_samples(self, ref):
        '''
        Get tau translation pairs (s1,s2) from the translations indexed by the reference translation.
        A pair (s1,s2) is in the output if BLEU(s1) - BLEU(s2) > alpha
        '''

        sys.stderr.write('Taking samples\n')
        sample = []
        best_candidates = self.data[ref] # get the dictionary of relevant translations

        for i in range(self.tau):
            # pick two translations from the dictionary at random
            a = randint(0, len(best_candidates.keys())-1 )   #subtract one because range is inclusive
            b = randint(0, len(best_candidates.keys())-1 )
            t1 = best_candidates.keys()[a]
            t2 = best_candidates.keys()[b]

            diff = individual_bleu(ref, t1, self.pseudocount) - individual_bleu(ref, t2, self.pseudocount)
            if (diff > self.alpha):
                sample.append((t1, t2, diff))
            elif (diff < -self.alpha):
                sample.append((t2, t1, diff))
            else:
                continue
        sample.sort(key=lambda a: math.fabs(a[2])) #in ascending order
        sys.stderr.write('Samples done\n')
        return sample[-self.sample_number:] 


    def score(self,translation_vector):
        '''
        Given a translation, return its score according to the linear model with the current weights
        '''
        score = 0
        for f in self.feats:
            score += self.weights[f] * float(translation_vector[f])
        return score, translation_vector

    def learn_weights(self):
        for i in range(self.epochs):
            sys.stderr.write('EPOCH '+str(i)+'\n')
            for r, ref in enumerate(self.references):
                sys.stderr.write('EPOCH '+str(i)+ ' Ref: '+str(r)+'\n')
                samples = self.get_samples(ref)
                self.perceptron_update(samples, ref)
	    return self.weights

    
    def perceptron_update(self, samples, ref):
        #do learning
        #update self.weights
        for s, samp in enumerate(samples):
            
            sys.stderr.write('Percentron update '+str(s)+'\n')
            s1 = samp[0]
            s2 = samp[1]
            for i, f in enumerate(self.feats):
                s1_feats = self.data[ref][s1].features
                s2_feats = self.data[ref][s2].features
                #pdb.set_trace()
                if self.weights[f] * s1_feats[f] <= self.weights[f] * s2_feats[f]:
                    gradient = 0
                    for feat in self.feats:                
                        gradient += self.eta * (s1_feats[feat] - s2_feats[feat]) # this is vector addition!
                    self.weights[f] = gradient

    def rank_and_bleu(self, out='pro_best_trans.out', learnt_weights=None):
        outfile = open(out,'w')
        
        if learnt_weights is not None: self.weights = learnt_weights 
        #assert len(learnt_weights.keys()) == len(self.feats)
        #TODO: make sure you only update the weights if the size is correct

        for r, ref in enumerate(self.references):
            best, best_t = (-1e300, '')
            for t in self.translations[r]:
                t_vec = self.data[ref][t].features
                curr_score, translation_vector = self.score(t_vec)

                if curr_score > best:
                    (best, best_t) = (curr_score, t)
            
            outfile.write(best_t+'\n')

        outfile.close()
        b = compute_bleu(out)
        return out , b

if __name__ == "__main__":
    pro_train = PRO()
    sys.stderr.write('Train Pro has been initialized\n')
    #TODO: need to see what's working... you can always run the tests in test_challenge.py
    #TODO: see why you're getting BLEU=0 for the 'train' set. Are the refs being picked up correctly? Probably not    
    
    #### TRAIN ####
    #rank and score with basic weights
    out_train, bleu_1_train = pro_train.rank_and_bleu()
    sys.stderr.write('Train 1 BLEU:' + str(bleu_1_train) + '\n')
    #learn the better weights
    learnt_weights = pro_train.learn_weights()
    sys.stderr.write('Weights learnt with perceptron\n')
    #rank and score with learnt weights
    out_train, bleu_2_train = pro_train.rank_and_bleu(weights=learnt_weights)
    sys.stderr.write('Train 2 BLEU:'+bleu_2_train+'\n')

    #assert bleu_2_train > bleu_1_train

    #### TEST ####
    #test the new weights on the test sets
    pro_test = PRO(train_location='dev+test/')
    sys.stderr.write('Test Pro has been initialized\n')

    out, bleu_1_test = pro_test.rank_and_bleu()
    sys.stderr.write('Train 1 BLEU:'+bleu_1_test+'\n')

    out, bleu_2_test = pro_test.rank_and_bleu(weights=learnt_weights)
    sys.stderr.write('Test 2 BLEU:'+bleu_2_test+'\n')



