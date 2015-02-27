from rerankfun import *
from compute_bleu_function import *
from collections import defaultdict
import os
import sys

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
'''
class PRO(train_data, test_data):
    self.train_data = train_data
    self.test_data = test_data
'''
def build_features(dataset='train',num_data=400,tps=100,norm=True):
    '''
    Given a dataset, build feature vectors for each of its translations.
    Requirements: a directory with the name dataset, which contains folders called 'norm' and 'unnorm'    
    
    @dataset either 'train' or 'dev+test' (the name of any directory containing data in the correct format)
    @num_data the number of data in the dataset
    @norm True if we want to use the normalised data, False otherwise
    @tps the number of translations per sentence, usually 100

    @return a list of lists of feature vectors. Each Russian sentence has a list of feature vectors, one for each 
    of its translations.
    '''
    normalised = '/norm' if norm else '/unnorm'
    feature_locations = os.listdir(dataset+normalised)
    print feature_locations
    features = []
    for location in feature_locations:
        feature_file = open(dataset+normalised+'/'+location,'r')
        feature = [line.strip() for line in feature_file]
        print len(feature), location
        assert len(feature) == num_data * tps
        features.append(feature)
        
    all_features = []
    for i in range(num_data):
        all_features.append([])      
        for j in range(i*tps, i*tps + tps):
            feat_vec = tuple([float(feature[j]) for feature in features])
            all_features[i].append(feat_vec)
    return all_features

def structure(dataset='train',tps=100):
    unstructured = [line.strip() for line in open(dataset+'/100best_clean.out','r')]
    structured = [[unstructured[j] for j in range(i*tps, (i+1)*tps)] for i in range(len(unstructured)/tps)]
    return structured

translations_list = structure(dataset='dev+test')
scores = build_features(dataset='dev+test', num_data=800, norm=False)
weights = [1,-1,1]
outname = 'dev+test/delete.out'
rerank(translations_list, scores, weights, outname)
print compute_bleu(outname)

def collect_bleu_scores():
    pass





