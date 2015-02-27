from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import os
import sys
from namedlist import namedlist

feat_norm = 'data/train/norm/'
feat_unnorm = 'data/train/unnorm/'
features = ['diag','ibm','lex','tm','diag_rev','ibm_rev','lm','untranslated']

##############################
#       Challenge
##############################
'''
We should use the rerankfun.rerank function, and pass in the following: 
rerank(sentences, scores, weights, outname)
'''

class PRO(object):

    def __init__(self,
                  train_location='train/', 
                  test_location='dev+test/', 
                  num_data=400,
                  tps=100
                  tau=5000,
                  alpha=0.1,
                  sample_number=100
                  eta=0.1
                  epochs=5):

        self.train_location = train_location
        self.test_location = test_location
        self.tps = tps
        self.num_train = 40000
        self.num_test = 80000
        self.tau = tau
        self.alpha = alpha
        self.sample_number = sample_number
        self.eta = eta
        self.epochs = epochs

        # load the training data: translations, references and source
        self.data = build_data()
        data_point = namedlist('blue', 'features')

    def build_data(self):
        '''
        Builds a dictionary of dictionaries. The 1st is indexed with a russian sentence. The 2nd with each candidate translation for that reference. 
        The value of the second dictionary is a data_point namedlist, which contains the blue score and the feature_vector values for the translation.
        '''
        self.data = defaultdict()
        references_trn = open(self.train_location+'train.ref','r')
        source_trn = open(self.train_location+'src.out','r')

        translations = self.structure(self.train_location)

        feature_vectors = self.build_features(self.train_location, self.num_train)
        bleu_list = self.collect_bleu_scores()

        for r, ref in enumerate(references_trn):
            self.data[ref] = defaultdict()
            for t, trans in enumerate(translations[r]):
                blue_score = bleu_list[r][t]
                feature_vec = feature_vectors[r][t]
                self.data[ref][trans] = data_point(blue_score, feature_vec)
                
            
            

    def build_features(self, data_location, num_data, norm=True):
        '''
        Given a dataset, build feature vectors for each of its translations.
        Requirements: a directory with the name dataset, which contains folders called 'norm' and 'unnorm'    
        
        @data_location either 'train' or 'dev+test' (the name of any directory containing data in the correct format)
        @num_data the number of data in the dataset
        @norm True if we want to use the normalised data, False otherwise

        @return a list of lists of feature vectors. Each Russian sentence has a list of feature vectors, one for each 
        of its translations.
        '''
        normalised = '/norm' if norm else '/unnorm'
        feature_locations = os.listdir(data_location+normalised)
        print feature_locations
        features = []
        for location in feature_locations:
            feature_file = open(data_location+normalised+'/'+location,'r')
            feature = [line.strip() for line in feature_file]
            print len(feature), location
            assert len(feature) == num_data * self.tps
            features.append(feature)
            
        all_features = []
        for i in range(num_data):
            all_features.append([])      
            for j in range(i*self.tps, i*self.tps + self.tps):
                feat_vec = tuple([float(feature[j]) for feature in features])
                all_features[i].append(feat_vec)
        return all_features

    def structure(self, data_location):
        '''
        Convert the unstructured datasets which aren't grouped by Russian sentence to a structured list, indexed by Russian sentence.
        '''
        unstructured = [line.strip() for line in open(data_location+'/100best_clean.out','r')]
        structured = [[unstructured[j] for j in range(i*self.tps, (i+1)*self.tps)] for i in range(len(unstructured)/self.tps)]
        return structured

    def collect_bleu_scores(self):
        '''
        Compute smoothed BLEU scores for each translation in the list of translations
        @translations_list a structured list of lists of translations, one list for each Russian sentence
        @return the BLEU scores in the same format, i.e. a list of lists
        '''
        bleu_list = [[0]*self.tps for _ in self.num_data]
        for rus_ind, translations in enumerate(translations_list):
            ref = references[rus_ind]
            for tr_ind, translation in enumerate(translation):
                bleu_score = individual_bleu(translation, ref)
                bleu_list[rus_ind].append(bleu_score)
        return bleu_list
         
    def bleu_difference(self,s1, s2, reference, alpha):
        stats_s1 = bleu.bleu_stats(s1, reference)
        stats_s2 = bleu.bleu_stats(s2, reference)

        s1_score = bleu.bleu(stats_s1)
        s2_score = bleu.bleu(stats_s2)

        return fabs(s1_score-s2_score)


    def getSamples(self, ref):
        '''
        Get tau translation pairs (s1,s2) from the translations indexed by the reference translation.
        A pair (s1,s2) is in the output if BLEU(s1) - BLEU(s2) > alpha, a certain threshold.
        '''
        sample = []
        best_candidates = self.data[ref] # get the dictionary of relevant translations
        for i in range(self.tau):
            # pick two translations from the dictionary at random
            t1, t2 = rnd.sample(best_candidates.keys(),2)
            diff = individual_bleu(t1, ref) - individual_bleu(t2, ref)
            if diff > self.alpha):
                sample.append(t1, t2, diff)
            elif diff < -self.alpha)::
                sample.append(t2, t1)
            else:
                continue
        sample.sort(key=lambda a: math.fabs(a[2])) #in ascending order
        return sample[-self.sample_number:] 


    def score(self,translation)
        '''
        Given a translation, return its score according to the linear model with the current weights
        '''
        score = 0
        for i in range(num_features):
            score += weights[i] * translation.features[i]
        return score

    

 
    #def perceptron(self, theta, s1, s2):
        

do a perceptron update of the parameters theta:
   if theta * s1.features <= theta * s2.features:
       mistakes += 1
       theta += eta * (s1.features - s2.features) # this is vector addition!



''' Test
translations_list = structure(dataset='dev+test')
scores = build_features(dataset='dev+test', num_data=800, norm=False)
weights = [1,-1,1]
outname = 'dev+test/delete.out'
rerank(translations_list, scores, weights, outname)
print compute_bleu(outname)

'''


