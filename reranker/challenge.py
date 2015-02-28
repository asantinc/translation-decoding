from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
from bleu import *
import os
import sys
from namedlist import namedlist
import pdb
from math import fabs


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
                  weights=[1.,1.,1.]):
        
        self.train_location = train_location
        self.tps = tps
        self.num_train = 800
        self.num_test = 800
        self.tau = tau
        
        self.alpha = alpha
        self.sample_number = sample_number
        self.eta = eta
        self.weights = weights
        self.num_feats=len(self.weights)
        self.epochs = epochs
        self.pseudocount = 0.0
        self.references = None

	    # Build up an ordered list of references
        self.references = self.build_ordered_list(self.train_location+'ref.out')
        
        #Build data array, that contains the translations per reference, with their scores and features
        self.data = self.build_data()
        self.build_features(self.num_train, norm=False)
        self.collect_bleu_scores(write=False)   


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
        feature_locations = [self.train_location + normalised + loc for loc in os.listdir(self.train_location+normalised)]

        features = []
        for location in feature_locations:
            feature = self.build_ordered_list(location)
            assert len(feature) == num_data * self.tps
            feature = self.structure(feature)
            features.append(feature)
        
        for i,ref in enumerate(self.references):
            for j, t in enumerate(self.translations[i]):
                feat_vec = tuple([float(feature[i][j]) for feature in features])
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
        sample = []
        best_candidates = self.data[ref] # get the dictionary of relevant translations
        for i in range(self.tau):
            # pick two translations from the dictionary at random
            t1, t2 = rnd.sample(best_candidates.keys(),2)
            diff = individual_bleu(ref, t1, self.pseudocount) - individual_bleu(ref, t2, self.pseudocount)
            if (diff > self.alpha):
                sample.append(t1, t2, diff)
            elif (diff < -self.alpha):
                sample.append(t2, t1, diff)
            else:
                continue
        sample.sort(key=lambda a: math.fabs(a[2])) #in ascending order
        return sample[-self.sample_number:] 


    def score(self,translation_vector):
        '''
        Given a translation, return its score according to the linear model with the current weights
        '''
        score = 0
        for i in range(self.num_feats):
            score += self.weights[i] * float(translation_vector[i])
        return score, translation_vector

    def learn_weights(self):
        for i in range(self.epochs):
            for ref in self.references:
                samples = self.get_samples(ref)
                self.perceptron_update(samples, ref)
	    return self.weights

    
    def perceptron_update(self, samples, ref):
        #do learning
        #update self.weights
        for samp in samples:
            s1 = samp[0]
            s2 = samp[1]
            for i, w in enumerate(self.weights):
                s1_feats = self.data[ref][s1].features
                s2_feats = self.data[ref][s2].features
                
                if w * s1_feat[i] <= w * s2_feat[i]:
                    gradient = 0
                    for j in self.weights:                
                        gradient += self.eta * (s1_feat[j] - s2_feat[j]) # this is vector addition!
                    self.weights[i] = gradient

    def rank_and_bleu(self, out='pro_best_trans.out'):
        outfile = open(out,'w')

        for r, ref in enumerate(self.references):
            best, best_t = (-1e300, '')
            for t in self.translations[r]:
                t_vec = self.data[ref][t].features
                curr_score, translation_vector = self.score(t_vec)

                if curr_score > best:
                    (best, best_t) = (curr_score, t)
            
            outfile.write(best_t+'\n')

        outfile.close()
        b = compute_bleu(temp_file)
        sys.stderr.write('BLEU: '+str(b))

if __name__ == "__main__":
    #weights = (('lex',1.), ('lm', 1.), ('tm', 1.))
    #pro_train = PRO(weights)
    pro_train = PRO()

    #check the output before training
    pro_train.rank_and_bleu()
    #learn the better weights
    learnt_weights = pro_train.learn_weights()
    pro_train.rank_and_bleu(weights=learnt_weights)

    #test the new weights on the test sets
    pro_test = PRO(train_location='dev+test/', weights=learnt_weights)
    pro_test.rank_and_bleu(weights=learnt_weights)



