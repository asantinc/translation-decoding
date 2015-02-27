from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import os
import sys
import pdb
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

class PRO(train_location='train/', test_location='dev+test/', num_data=400,tps=100):

    def __init__(self):

        self.train_location = train_location
        self.test_location = test_location
        self.tps = tps
        self.num_train = 40000
        self.num_test = 80000

        # load the training data: translations, references and source
        translations_trn = self.structure(train_location)
        references_trn = open(train_location+'train.ref','r')
        source_trn = open(train_location+'src.out','r')

        # build the feature vectors for each translation
        scores_trn = self.build_features(train_location, num_train)

        # collect the BLEU scores for each translation
        bleu_list = self.collect_bleu_scores()

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

