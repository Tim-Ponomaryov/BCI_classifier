import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from random import choice
from itertools import product

from eeg_dataset_utils import EEGDataset, EEGDatasetAdvanced, flatten

GROUP1 = [(0, 11, 21), (1, 12, 22), (2, 13, 23),
        (3, 14, 24), (4, 15, 25), (5, 16, 26),
        (6, 17, 18), (7, 9, 19), (8, 10, 20)]
GROUP2 = [(0, 17, 24), (1, 9, 25), (2, 10, 26), 
        (3, 11, 18), (4, 12, 19), (5, 13, 20), 
        (6, 14, 21), (7, 15, 22), (8, 16, 23)]

class OfflineBCI():
    
    def __init__(self, test_set, model, model_type='ML', use_heuristic=False):
        '''
        Class for offline BCI performance estimation
        
        Arguments:
        test_set -- EEGDataset or EEGAdvancedDataset with test data
        model -- ML model from sklearn of NN from pytorch
        model_type -- str 'ML' or 'NN'
        
        '''
        
        self.dataset = test_set
        self.info = test_set.info.copy()
        self.info.reset_index(inplace=True, names='prev_index')
        self.model = model
        self.model_type = model_type 
        
        self.groups = GROUP1+GROUP2 # Groups of stimuli on the screen; contain symbol indecies in self.stims
        self.stims = [item for item in u'qwertyuiopasdfghjklzxcvbnm_1234567890!?.,;:"()+=-~[]\/']
        
        self.N = 27 # Number of possible commands -> 27 symbols
        self.P = None # Classification accuracy -> n correct commands/total commands
        self.T = None # Timing of a BCI operation -> 18*(stim+isi)*n_cycles
        self.stim_time = 0.025
        self.isi = 0.075
        
        self.result = {
            'total_trials': 0,
            'correct_trials': 0,
            'target_letter': [],
            'guess': []
        }
        
        self.use_heuristic = use_heuristic
            
    def average(self, data:dict):
        '''Average epochs for similar groups of stimuli'''
        
        x = []        
        for k, v in data.items():
            if len(v)==1:
                x.append(v[0])
            else:
                x.append(np.average(v, axis=0))
        
        return np.stack(x)
    
    def get_timing(self, cycles:int):
        '''
        Calculates timing that is required to type a command, sec
        and write it into self.T
        
        '''
        
        self.T = 18*(self.stim_time+self.isi)*cycles
        
    
    def choose_groups(self, X):
        '''Run model and choose groups'''
        
        # Make predictions
        if self.model_type=='ML':
            out = self.model.predict(flatten(X))
        else:
            self.model.eval()
            X = torch.tensor(X)
            out = self.model(X).detach().numpy().reshape(18)
        
        # Find target predicions
        ser = pd.Series(out)
        group_ids = ser[ser<=0.5].index.to_list()
        
        # Devide predictions into groups according to given groups
        gr1_ids = list(filter(lambda x: x<9, group_ids.copy()))
        gr2_ids = list(filter(lambda x: x>=9, group_ids.copy()))
        
        if (0 == len(gr1_ids)) or (0 == len(gr2_ids)):
            # When no group is classified as target
            gr1 = gr2 = None
        elif (len(gr1_ids) > 1) or (len(gr2_ids) > 1):
            # When classification is ambigueus
            if self.use_heuristic:
                gr1, gr2 = self.heuristic(gr1_ids, gr2_ids)
            else:
                gr1 = gr2 = None
        else:
            # When classification is accurate
            gr1 = gr1_ids[0]
            gr2 = gr2_ids[0]

        if isinstance(gr1, int) and isinstance(gr2, int):
            return self.groups[gr1], self.groups[gr2]    
        
        return gr1, gr2
        
    
    def heuristic(self, gr1_ids, gr2_ids):
        '''Try to find possible correct guesses from given'''
        
        comb = list(product(gr1_ids, gr2_ids))
        possible = [] # Container to store possibly true answers
        for g1, g2 in comb:
            gr1 = set(self.groups[g1])
            gr2 = set(self.groups[g2])
            intersection = gr1.intersection(gr2)
            if len(intersection)==1:
                possible.append((g1, g2))
                # return g1, g2
            
        if len(possible)>=1:
            gr1, gr2 = choice(possible)
        elif len(possible)==0:
            gr1 = gr2 = None
        
        return gr1, gr2
            
    
    def get_letter(self, gr1:tuple, gr2:tuple) -> (int, str):
        '''Get letter at the interseption of 2 groups
        
        Returns:
        (index: int, letter: str)
        
        '''
        
        gr1 = set(gr1)
        gr2 = set(gr2)
        intersection = gr1.intersection(gr2)
        guess = 'Fail' if len(intersection)!=1 else self.stims[intersection.pop()]
        
        return guess
    
    def calculate_P(self):
        
        self.P = self.result['correct_trials']/self.result['total_trials']
    
    def calculate_ITR(self):
        '''Calculates information transfer rate: bit/min'''
        
        if self.P==0:
            return 0

        if self.P==1:
            self.P=0.99
            
        return(60*(self.P*np.log2(self.P)+(1-self.P)*np.log2((1-self.P)/(self.N-1))+np.log2(self.N)))/self.T
    
    def process_trial(self, info):
        '''Process trial using info
        
        Arguments:
        info -- info for a specific trial
        
        '''
        # Get epochs for each group of stimuli
        data = dict([k, []] for k in range(18))
        for idx, gr in info.iterrows():            
            data[gr.code].append(self.dataset[idx][0])
        # Average across each gropup
        X = self.average(data)
        gr1, gr2 = self.choose_groups(X)
        if isinstance(gr1, tuple) and isinstance(gr2, tuple):
            guess = self.get_letter(gr1, gr2)
        else:
            # If cannot choose epochs -> autofail
            guess = 'Fail'
        
        return guess
    
    def summary(self):
        '''Print summary on classification'''
        
        print(f"Total trials: {self.result['total_trials']}")
        print(f"Correct trials: {self.result['correct_trials']}")
        print(f'ITR: {self.itr:.2f}')
        print(f'Classification accuracy: {self.P:.2f}')
        
    
    def pipeline(self, n, summary=False):
        '''
        Function to emulate BCI process.
        Needs to pass epochs in classifier, estimate score for
        each stimulus group and than choose a letter to type.  
        
        Arguments:
        n -- n epochs will be averaged for each group of stimuli
        
        Returns:
        itr -- information transfer rate, bit/min
        P -- classification accuracy
        
        '''
        
        self.get_timing(n)
        # indicies of the last letters in a row of similar letters
        ids = np.where(self.info.target_letter[:-1].values != self.info.target_letter[1:].values)[0] + 1
        idx = self.info.index.values # epochs indicies
        splidx = np.split(idx, ids) # list of indicies corresponding to each target letter
        info_by_letter = [self.info.loc[i] for i in splidx] # splitted version of info corresponding each target letter
        
        for info in info_by_letter:
            letter = info.target_letter.values[0] # target letter in the current run
            trial_id = np.arange(len(info)+1)[::18*n][1:-1] # ids to separate epochs info by trials -- multiple of 18 epochs
            splidx = np.split(info.index.values, trial_id) # ids split info into trials (batch) info
            trial_info = [info.loc[i] for i in splidx] # list of info for each trial
            for ti in trial_info:
                guess = self.process_trial(ti)
                # check if the guess is correct
                if guess == letter:
                    self.result['correct_trials'] += 1
                self.result['total_trials'] += 1
                self.result['target_letter'].append(letter)
                self.result['guess'].append(guess)
        
        self.calculate_P()
        self.itr = self.calculate_ITR()
        
        if summary:
            self.summary()
        
        return self.itr, self.P
    
    
from copy import deepcopy

def split_by_words(dataset, n:int):
    '''Split the dataset into 2 including specific number of words (5 letters)
    
    Arguments:
    n -- n words to be included in train_set
    
    '''
    
    assert 0 < n < dataset.info.word_n.unique().max()+1, 'n must be 0 < n < n_words'
    
    ids_train = dataset.info[dataset.info.word_n < n].index
    ids_test = dataset.info[dataset.info.word_n >= n].index
    
    if dataset.__class__.__name__ == 'EEGDataset':
        train_set = EEGDataset(root_dir=dataset.root_dir, 
                           data=dataset.x[ids_train,:,:],
                           labels=dataset.y[ids_train],
                           info=dataset.info.iloc[ids_train])
        test_set = EEGDataset(root_dir=dataset.root_dir,
                            data=dataset.x[ids_test,:,:],
                            labels=dataset.y[ids_test],
                            info=dataset.info.iloc[ids_test])
    else:
        train_set = deepcopy(dataset)
        test_set = deepcopy(dataset)
        
        train_set.data = pd.Series(dataset.data)[ids_train].values.tolist()
        test_set.data = pd.Series(dataset.data)[ids_test].values.tolist()
        train_set.info = dataset.info.iloc[ids_train]
        test_set.info = test_set.info.iloc[ids_test]
    
    return train_set, test_set


def itr(P:float, N:int, T:float) -> float:
    '''ITR calculator
    
    Arguments:
    P -- classification accuracy (correct trials/total trials);
    N -- number of possible commands;
    T -- time of each trial in seconds;
    '''
    
    return (60*(P*np.log2(P)+(1-P)*np.log2((1-P)/(N-1))+np.log2(N)))/T