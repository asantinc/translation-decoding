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
                  xi=100,
                  eta=0.1,
                  epochs=20,
                  weights=None, write_bleu=False, norm=True):
        
        self.train_location = train_location
        
        self.tps = tps
        if train_location == 'train/':
            self.num_train = 400
        else:
            self.num_train = 800
            

        #perceptron training variables
        self.tau = tau
        self.alpha = alpha
        self.xi = xi
        self.eta = eta
        self.epochs = epochs
        self.pseudocount = 0.0

        if weights is None: weights = { 'lm':1. , 'tm':1., 'lex': 1.}  
        self.weights = weights
        self.feats = self.weights.keys()

	    # Build up an ordered list of references
        self.references = self.build_ordered_list(self.train_location+'ref.out')
        
        #Build data array, that contains the translations per reference, with their scores and features
        self.data = self.build_data()
        self.build_features(self.num_train, norm=norm)
        self.collect_bleu_scores(write_bleu)   


    #TODO: What order do the weights appear in?
    def set_weights(self, weight_vector):
        '''
        Update the class weights values to a different set of weights, if their lengths match
        '''
        assert len(weight_vector) == len(self.weights)
        self.weights = weight_vector


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
            print num_data * self.tps
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


    def build_ordered_list(self, filename):    
        '''
        Opens a file, produces a list with lines, and closes the file. This helps ensure that files close.
        '''
        f = open(filename,'r')
        ordered = [line.rstrip('\n') for line in f]
        f.close()
        return ordered
        

    def structure(self, dataset):
        '''
        Convert the unstructured datasets which aren't grouped by Russian sentence to a structured list, indexed by Russian sentence.
        '''
        unstructured = dataset
        structured = [[unstructured[j] for j in range(i*self.tps, (i+1)*self.tps)] for i in range(len(unstructured)/self.tps)]
        return structured


    def collect_bleu_scores(self,write=True):
        '''
        Compute smoothed BLEU scores for each translation in the list of translations
        @translations_list a structured list of lists of translations, one list for each Russian sentence
        @return the BLEU scores in the same format, i.e. a list of lists
        '''
        if write:
            outfile = open(self.train_location+'bleu_scores','w')
            for ref in self.references:
                translations = self.data[ref]
                for t in translations:
                    bleu_score = individual_bleu(t, ref)
                    translations[t].bleu = bleu_score
                    outfile.write(str(bleu_score)+'\n')
            outfile.close()
        else:
            bleu_scores_unstruct = self.build_ordered_list(self.train_location+'bleu_scores')
            bleu_scores_list = self.structure(bleu_scores_unstruct)
            for i, ref in enumerate(self.references):
                for j, t in enumerate(self.data[ref]):
                    bleu_score = bleu_scores_list[i][j]
                    self.data[ref][t].bleu = float(bleu_score)
        
          
    def score(self,translation_vector):
        '''
        Given a translation, return its score according to the linear model with the current weights
        '''
        score = 0
        for f in self.feats:
            score += self.weights[f] * float(translation_vector[f])
        return score, translation_vector

	
    def learn_weights(self):
        '''
        Learn new weights by:
        1. Taking samples of pairs of translations for each group of translated sentences
        2. Updating the feature weights based on who wins in terms of BLEU scores vs. which of their individual features*weights beat each other
        '''
        for i in range(self.epochs):
            self.mistakes = 0
            sys.stderr.write('--------- Epoch '+str(i)+'--------- \n')
            for r, ref in enumerate(self.references): 
                #sys.stderr.write('EPOCH '+str(i)+ ' Ref: '+str(r)+'/'+str(len(self.references))+'\n')
                samples = self.get_samples(ref)
                self.perceptron_update(samples, ref)
            # Sanity checking: do the mistakes go down? Does BLEU go up?
            sys.stderr.write('Mistakes: '+str(self.mistakes)+'\n')  
            out, b = self.rank_and_bleu()  
            sys.stderr.write('BLEU score: '+str(b)+'\n')  
            sys.stderr.write(str(self.weights)+'\n')  
        return self.weights


    def get_samples(self, ref):
        '''
        Get tau translation pairs (s1,s2) from the translations indexed by the reference translation.
        A pair (s1,s2) is in the output if BLEU(s1) - BLEU(s2) > alpha
        '''

        #sys.stderr.write('Taking samples\n')
        sample = []

        for i in range(self.tau):
            # pick two translations from the dictionary at random
            a = randint(0, self.tps-1 )   #subtract one because range is inclusive
            b = randint(0, self.tps-1 )
            translations = self.data[ref].keys()
            t1 = translations[a]
            t2 = translations[b]

            diff = self.data[ref][t1].bleu - self.data[ref][t2].bleu
            if (diff > self.alpha):
                sample.append((t1, t2, diff))
            elif (diff < -self.alpha):
                sample.append(( t2, t1, fabs(diff) ))
            else:
                continue
        sample.sort(key=lambda a: math.fabs(a[2])) #in ascending order
        #sys.stderr.write('Samples done\n')
        return sample[-self.xi:] 

    
    def perceptron_update(self, xi_samples, ref):
        '''
        Play sentences against each other for each of their features
        '''
        for s, samp in enumerate(xi_samples):
            s1 = samp[0]
            s2 = samp[1]

            #for i, f in enumerate(self.feats):  #for each feature pair of the samples
            s1_feats = self.data[ref][s1].features
            s2_feats = self.data[ref][s2].features
            assert self.data[ref][s1].bleu > self.data[ref][s2].bleu    #s1 should always have the better BLEU score

            #calculate their scores based on their features (w*features)
            s1_feats_score = self.dot_product(s1_feats)
            s2_feats_score = self.dot_product(s2_feats)
            
            if s1_feats_score <= s2_feats_score:        #if s1 ranks lower even if my BLEU score is higher, update the weights!
                self.mistakes += 1
                for feat in self.feats:         
                    #sys.stderr.write(str(self.weights)+'\n') 
                    self.weights[feat] += self.eta * (s1_feats[feat] - s2_feats[feat]) 
                    #sys.stderr.write(str(self.weights)+'\n') 
        #pdb.set_trace()             
    
    def dot_product(self, feat_vec):    
        '''
        Return dot product between the weights vector and a given feature vector
        '''    
        dot_prod = 0
        for feat, weight in self.weights.iteritems():
            dot_prod += weight*feat_vec[feat]
        return dot_prod


    def rank_and_bleu(self, out='pro_best_trans.out', weights=None):
        '''
        Rank translations for every reference and output the top translation to file. 
        Score the chosen translations using BLEU and return the BLEU score
        '''
        outfile = open(out,'w')
        
        if weights is not None: self.weights = weights 
        #assert len(learnt_weights.keys()) == len(self.feats)

        for r, ref in enumerate(self.references):
            best, best_t = (-1e300, '')
            for t in self.translations[r]:
                t_vec = self.data[ref][t].features
                curr_score, translation_vector = self.score(t_vec)

                if curr_score > best:
                    (best, best_t) = (curr_score, t)
            
            outfile.write(best_t+'\n')

        outfile.close()
        b = compute_bleu(out, ref=self.train_location+'ref.out')
        return out , b

if __name__ == "__main__":

    #### TRAIN ####
    #original_weights = {'lm':10., 'lex':10., 'tm':10., 'diag':10., 'ibm':10.}
    original_weights = {'lm':0., 'lex':0., 'tm':0., 'bias':0., 'ibm':0., 'diag':0}
    temp_weights = original_weights.copy()
    pro_train = PRO(weights= original_weights, norm=True)
    sys.stderr.write('\n ******Train Pro has been initialized****** \n')
    sys.stderr.write('Weights original : '+str(original_weights)+'\n')
    
    #rank and score with original weights
    out_train, bleu_original_train = pro_train.rank_and_bleu(weights=original_weights)
    sys.stderr.write('Train Original BLEU:' + str(bleu_original_train) + '\n')

    #learn the better weights
    learnt_weights = pro_train.learn_weights()
    sys.stderr.write('Weights learnt   : '+str(learnt_weights)+'\n')

    #rank and score with learnt weights with the training
    out_train, bleu_learnt_train = pro_train.rank_and_bleu(weights=learnt_weights)
    sys.stderr.write('Train Original BLEU :' + str(bleu_original_train) + '\n')
    sys.stderr.write('Train Learnt BLEU   :'+str(bleu_learnt_train)+'\n')
    #assert bleu_2_train > bleu_1_train

    #### TEST ####
    #test the new weights on the test sets
    pro_test = PRO(train_location='dev+test/')
    sys.stderr.write('\n *******Test Pro has been initialized*********\n')

    out, bleu_original_test = pro_test.rank_and_bleu(weights=temp_weights)
    sys.stderr.write('Test Original BLEU:'+str(bleu_original_test)+'\n')

    out, bleu_learnt_test = pro_test.rank_and_bleu(weights=learnt_weights)
    sys.stderr.write('Test Learnt BLEU:'+str(bleu_learnt_test)+'\n')



