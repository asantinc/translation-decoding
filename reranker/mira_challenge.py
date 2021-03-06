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
                  xi=1000,
                  C=1000,
                  epochs=5,
                  weights=None, write_bleu=False ):
        
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
        self.C = C
        self.epochs = epochs
        self.pseudocount = 0.0

        if weights is None: weights = { 'lm':1. , 'tm':1., 'lex': 1.}  
        self.weights = weights
        self.feats = self.weights.keys()

	    # Build up an ordered list of references
        self.references = self.build_ordered_list(self.train_location+'ref.out')
        oracle_translations = self.build_ordered_list(self.train_location+'oracle.out')
        oracle_bleu = self.build_ordered_list(self.train_location+'oracle_bleu.out')
        self.oracle = {r:t for r,t in zip(self.references, oracle_translations)}
        
        #Build data array, that contains the translations per reference, with their scores and features
        self.data = self.build_data()
        self.build_features(self.num_train, norm=True)
        self.collect_bleu_scores(write_bleu)   


    #TODO: What order do the weights appear in?
    def set_weights(self, weight_vector):
        '''
        Update the class weights values to a different set of weights, if their lengths match
        '''
        assert len(weight_vector) == len(self.weights)
        self.weights = weight_vector


    def build_data(self):
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

	
    def learn_weights(self, tau=None, epoch=None, pseudocounts=1.):
        '''
        Learn new weights by:
        1. Taking samples of pairs of translations for each group of translated sentences
        2. Updating the feature weights based on who wins in terms of BLEU scores vs. which of their individual features*weights beat each other
        '''
        if tau is not None: self.tau = tau
        if epoch is not None: self.epochs = epoch
        self.pseudocount = pseudocounts

        for i in range(self.epochs):
            self.mistakes = 0
            sys.stderr.write('--------- Epoch '+str(i)+'--------- \n')
            sys.stderr.write('Weights: '+ str(self.weights))
            for r, ref in enumerate(self.references): 
                #sys.stderr.write('EPOCH '+str(i)+ ' Ref: '+str(r)+'/'+str(len(self.references))+'\n')
                self.update_weights(ref)
            
            # Sanity checking: do the mistakes go down? Does BLEU go up?
            sys.stderr.write('Mistakes: '+str(self.mistakes)+'\n')  
            out, b = self.rank_and_bleu()  
            sys.stderr.write('BLEU score: '+str(b)+'\n')  
        return self.weights

    def subtract(self,u,v):
        return tuple(float(ui - vi) for ui,vi in zip(u,v))
    
    def dot(self,u,v):
        return sum(1.0*ui*vi for ui,vi in zip(u,v))

    def norm_squared(self,u):
        return self.dot(u,u)

    def update_weights(self, ref):
        loss, t, diff_vec = self.mira_loss(ref)
        #t_vec = (self.data[ref][t].features[feat] for feat in self.feats)
        #oracle_vec = (self.data[ref][self.oracle[ref]].features[feat] for feat in self.feats)
        #diff_vec = self.subtract(oracle_vec, t_vec)
        assert type(diff_vec) is tuple
       
        D = self.norm_squared(diff_vec)
        eta = min(self.C, loss*1.0/D) if D != 0 else self.C
        #print eta
        for i, w in enumerate(self.weights.keys()):
            self.weights[w] += eta * -1.0 * diff_vec[i]


    def mira_loss(self, ref):
        '''
        Compute the MIRA loss for the current weight vector on a given reference sentence.
        '''
        translations = self.data[ref]  # get the translations
        oracle_t = self.oracle[ref]  # get the oracle translation
        current_loss = -1e300
        for t in translations:
            bleu_loss = self.data[ref][oracle_t].bleu - self.data[ref][t].bleu  # compute the difference in BLEU between this t and the best
            #model_loss = self.dot_product(self.data[ref][t].features) - self.dot_product(self.data[ref][oracle_t].features)  # compute model loss
            t_vec = (self.data[ref][t].features[feat] for feat in self.feats)
            oracle_vec = (self.data[ref][self.oracle[ref]].features[feat] for feat in self.feats)
            diff_vec = self.subtract(t_vec, oracle_vec)
            w_vec = (self.weights[feat] for feat in self.feats)
            model_loss = self.dot(w_vec, diff_vec)
            new_loss = bleu_loss + model_loss
            if new_loss > current_loss:
                (current_loss, worst_offender, diff_worst) = (new_loss, t, diff_vec)
        return current_loss, worst_offender, diff_worst
    
    
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
    original_weights = {'lm':1., 'lex':1., 'tm':1., 'ibm':1, 'diag':1}
    temp = original_weights.copy()
    pro_train = PRO(weights=original_weights)
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

    out, bleu_original_test = pro_test.rank_and_bleu(weights=temp)
    sys.stderr.write('Test Original BLEU:'+str(bleu_original_test)+'\n')

    out, bleu_learnt_test = pro_test.rank_and_bleu(weights=learnt_weights)
    sys.stderr.write('Test Learnt BLEU:'+str(bleu_learnt_test)+'\n')



