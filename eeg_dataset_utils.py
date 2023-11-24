import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from random import shuffle

import torch
from torch.utils.data import Dataset
from torch import nn as nn 
import torchvision.transforms.functional as F

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class EEGDataset(Dataset):
    '''Simple dataset for P300 eeg data for one subject'''
    
    def __init__(self, root_dir:str=None, subject:str=None, data:np.ndarray=None, labels:np.ndarray=None,
                 info:pd.DataFrame=None, transform=None, channels=[], average=None, downsample=None):
        
        super().__init__()
        
        self.root_dir = root_dir
        self.subject = subject
        # self._subjects() # list of all subject codes
        self.ch_names = self._channels() # list of channels
        self.transform = transform
        
        # Constructor from arrays
        if isinstance(data, np.ndarray) and isinstance(labels, np.ndarray):
            self.data = data
            self.x = data.copy()
            self.labels = labels
            self.y = labels.copy()
            self.info = info
        else:
            # Constructor from files
            self.load_data()
        
        # Labels decoding    
        self.labels_info = {0:'target', 1:'non-target'}
        
        # Data augmentations
        self.downsampling_coef = downsample
        self.n_average = average
        self.pick_channels(channels)
        
        self._adjust_data()
        
    def __len__(self):
        
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        
        x = self.x[idx,:,:]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def load_data(self):
        
        assert self.root_dir, 'Root directory is not specified'
        assert self.subject, 'Subject is not specified'
        
        self.data = np.load(os.path.join(self.root_dir, f'{self.subject}_c_epochs.npy'))
        self.x = self.data.copy()
        self.labels = np.load(os.path.join(self.root_dir, f'{self.subject}_c_labels.npy'))
        self.y = self.labels.copy()
        self.info = pd.read_csv(os.path.join(self.root_dir, 'info', f'{self.subject}_c_events_info.csv'), header=0, index_col=0)
        self.info.reset_index(inplace=True, names='old_index')
    
    def _channels(self):
        '''Get list of channels (electrodes) in data
        
        Reflects the 2d dimension of (n, 44, 301) in data
        
        '''
        
        # from ast import literal_eval
        # with open(os.path.join(self.root_dir, 'eeg_ch_names.txt'), 'r') as f:
        #     ch_names = literal_eval(f.readline())
        
        ch_names = ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                         'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                         'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                         'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                         'PO3', 'POz', 'PO4', 'O1', 'O2']
        
        return ch_names
            
    def _subjects(self):
        '''Get list of subjects of the data'''
        
        files = os.listdir(self.root_dir)
        files = list(filter(lambda x: x.endswith('.npy'), files))
        self.available_subjects = list(set(map(lambda x:x.split('_')[0], files)))
        self.available_subjects.sort()
                
    def pick_channels(self, channels:list=[]):
        '''Pick some eeg channels according to list of electordes
           given in channels argument
        
        '''
        
        assert isinstance(channels, list), 'Channels must be given as list'
        
        if channels==[]:
            self.picked = self.ch_names.copy()
            self.mask=None
        else:
            self.picked = channels
            self.mask = [self.ch_names.index(ch) for ch in channels]
            self.mask.sort()
        self._adjust_data()
    
    def _pick_channels(self):
        '''Pick eeg channels'''
        
        if self.mask:
            self.x = self.x[:,self.mask,:]
    
    def average(self, n:int):
        '''Calculate average of n channels for all epochs'''
        
        self.n_average = n
        self._adjust_data()
        
    def _average(self):
        '''Calculate average of n channels for all epochs'''
        
        x_t = self.x[self.y==0].copy()
        x_nt = self.x[self.y==1].copy()
        x_t_ave = self._average_one(x_t, self.n_average)
        x_nt_ave = self._average_one(x_nt, self.n_average)
        self.x = np.vstack([x_t_ave, x_nt_ave])
        self.y = np.hstack([np.zeros(x_t_ave.shape[0]), np.ones(x_nt_ave.shape[0])])
        
    def _average_one(self, x, n:int):
        '''Calculate average for n suиsequent samples in x'''
        
        x_dev = [x[n-i::n] for i in range(1,n+1)]
        min_shape = min([ar.shape for ar in x_dev])
        x_dev = [arr[1:]  if (arr.shape > min_shape) else arr for arr in x_dev]
        return np.mean(np.array(x_dev), axis=0)
    
    def downsample(self, n:int):
        '''Make downsampling on data'''
        
        self.downsampling_coef = n
        self._adjust_data()
    
    def _downsample(self):
        '''Make downsampling on data'''
        
        self.x = self.x[:,:,::self.downsampling_coef]
    
    def _adjust_data(self):
        '''Do changes on data'''
        
        self.x = self.data.copy()
        self.y = self.labels.copy()
        
        if self.downsampling_coef:
            self._downsample()
        if self.mask:
            self._pick_channels()
        if self.n_average:
            self._average()

def merge_datasets(*datasets) -> EEGDataset: 
    '''Merge eeg datasets in one'''
    
    data = np.vstack([d.data for d in datasets])
    labels = np.hstack([d.labels for d in datasets])
    info = pd.concat([d.info for d in datasets])
    
    return EEGDataset(data=data, labels=labels)

class EEGDatasetAdvanced(Dataset): 
    '''Advanced dataset to operate with wider amount of EEG data'''
    
    def __init__(self, root_dir:str=None, load_cache=True, discret=35, cache_dir_name:str='eeg_cache',
                 subjects:list=[], transform=None, n_average:int=None, reverse_labels=False, **kwargs):
        '''
        Keyword arguments:
        root_dir -- a directory with .npy files that contain epochs
        load_cache -- flag to load existing cache instead making new one
        discret -- cache discretization
        cache_dir_name -- name of directory for cache to save it to or load it from
        subjects -- names of subjects to load their data
        transform -- ...
        average -- int of how many epochs should average
        reverse_labels -- bool means whether to swipe class names (t:0, nt:1) -> (t:1, nt:0)
                          for better confusion matrix development
        
        **kwargs:
        downsampe -- int that means the downsampling factor when creating a new cache
        dtype -- numpy.dtype like np.float16 to convert data when creating a new cache
        
        '''
        
        super().__init__()
        
        self.root_dir = root_dir
        self.cache_path = os.path.join(self.root_dir, cache_dir_name)
        self.load_cache = load_cache
        self.subjects = subjects
        self.transform = transform
        self.discret = discret
        
        self.n_average=n_average
        
        self.reverse_labels=reverse_labels
        self.labels_info = {0:'target', 1:'non-target'}
        
        self.get_subjects() # list of available subjects
        self.get_channels() # EEG channels names
        self.load_data(**kwargs) # load data
        if self.subjects != []:
            self.pick_subjects(self.subjects)
        if self.n_average:
            self.average(n_average)
        self.get_info()
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.n_average:
            x, y = self._average(idx)
        else:
            x = torch.tensor(np.load(os.path.join(self.cache_path, self.data[idx])), dtype=torch.float32)
            y = torch.tensor(int(self.data[idx].split('_')[-2]), dtype=torch.float32)
        
        if self.mask: # Choose channels if needed
            x = x[self.mask, :]
        
        if self.transform:
            x = self.transform(x)
        
        if self.reverse_labels:
            y = np.abs(y-1)
        
        return x, y
    
    def average(self, n):
        '''Do averaging'''
        if self.n_average:
            self.load_data()
            self.pick_subjects(self.subjects)
        self.n_average = n
        self.average_list()
    
    def _average(self, idx):
        '''Average epochs'''
        
        y = torch.tensor(int(self.data[idx][0].split('_')[-2]), dtype=torch.float32)
        files = [os.path.join(self.cache_path, f) for f in self.data[idx]]
        x = np.average([np.load(f) for f in files], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        
        return x, y
    
    def average_list(self):
        '''Adjust data property to be able to average epochs'''
        
        data = self.data.copy()
        target = list(filter(lambda x: int(x.split('_')[-2])==0, data))
        nontarget = list(filter(lambda x: int(x.split('_')[-2])==1, data))
        data = []
        for lst in (target, nontarget):
            data += list(self.grouped(lst, self.n_average))
        
        self.data = data
        
    def load_data(self, **kwargs):
        
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if self.load_cache:
            # print('Cache is already available')
            self.data = os.listdir(self.cache_path)
            self.data.sort(key=lambda x: (x.split('_')[0], int(x.split('_')[2])))
            return
        self._load_and_cache(**kwargs)
                
    def _load_and_cache(self, downsample=1, dtype=np.float64, **kwargs):
        '''cycle across all subjects to save each epoch separately'''
        
        names = self.subjects if self.subjects else self.available_subjects        
        for name in tqdm(names):
            data = np.load(os.path.join(self.root_dir, f'{name}_c_epochs.npy'))
            labels = np.load(os.path.join(self.root_dir, f'{name}_c_labels.npy'))
            data = data.astype(dtype)[:,:,::downsample]
            for i in range(data.shape[0]):
                np.save(os.path.join(self.cache_path, f'{name}_epoch_{i}_class_{labels[i]}_.npy'), data[i])
        print(f'All data is cached in {self.cache_path}')
        self.data = os.listdir(self.cache_path)
    
    def get_subjects(self):
        '''Get list of subjects of the data'''

        files = os.listdir(self.cache_path) if self.load_cache else os.listdir(self.root_dir)
        files = list(filter(lambda x: x.endswith('.npy'), files))
        self.available_subjects = list(set(map(lambda x:x.split('_')[0], files)))
        self.available_subjects.sort()
        if self.subjects == []:
            self.subjects = self.available_subjects.copy()
    
    def get_info(self):
        
        if self.n_average:
            return
        
        info_list = []
        for subject in self.subjects:
            info = pd.read_csv(os.path.join(self.root_dir, 'info', f'{subject}_c_events_info.csv'), header=0, index_col=0)
            info.reset_index(inplace=True, names='old_index')
            info_list.append(info)
        self.info = pd.concat(info_list)
                
        
    def get_channels(self):
        '''Get list of channels (electrodes) in data
        
        Reflects the 2d dimension of (n, 44, 301) in data
        
        '''
        # Load from file:
        # from ast import literal_eval
        # with open('./P300BCI_DataSet/eeg_ch_names.txt', 'r') as f:
        #     self.ch_names = literal_eval(f.readline())
        
        # Given list
        self.ch_names = ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                         'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                         'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                         'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                         'PO3', 'POz', 'PO4', 'O1', 'O2']
        self.mask = None
                    
    def pick_channels(self, channels:list=[]):
        '''Pick some eeg channels according to list of electordes
           given in channels argument
        
        '''
        
        if channels==[]:
            
            self.mask = None
            self.picked_channels = self.ch_names
            return
        
        self.picked_channels = channels
        self.mask = [self.ch_names.index(ch) for ch in channels]
        
    def pick_subjects(self, subjects=[]):
        '''Pick given subjects from a dataset'''
        
        self.data = [file for file in self.data if file.split('_')[0] in subjects]
        self.subjects = subjects
        self.subjects.sort()
        self.data.sort(key=lambda x: (x.split('_')[0], int(x.split('_')[2])))
        if not self.n_average:
            self.get_info()
    
    @staticmethod    
    def grouped(iterable, n):
        '''Make chunks of n consecutive elements in iterable'''
        return zip(*[iter(iterable)]*n)

class Normalize(nn.Module):
    '''Normalization for non-image data
    
    A clone of torchvision.transforms.Normalize for non-image data
    
    '''
    
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def forward(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)

class Flatten(nn.Module):
    '''Flatten tensor across a given dimension'''
    
    def __init__(self, dim):
        
        super().__init__()
        self.dim = dim
        
    def forward(self, tensor):
        
        return tensor.flatten(self.dim)
    
class Unsqueeze(nn.Module):
    '''Insert a dimention in a given position'''
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, tensor):
        
        return tensor.unsqueeze(self.dim)

def flatten(array:np.ndarray):
    return array.reshape([array.shape[0], array.shape[1]*array.shape[2]])

def my_simple_split(dataset:EEGDataset, size:list):
    '''
    Splits the EEGDataset into train and test set
    
    Arguments
    dataset -- EEGDataset instance
    size -- a list len of 2 [train_size, test_size]
    
    Returns:
    tuple of train_set and test_set EEGDatasets
    
    '''
    
    assert len(size)==2, 'size must be given as [train_size, test_size] in parts [0.8, 0.2] e.g.'
    train_size, test_size = size
    
    idx_sep = round(len(dataset)*train_size)
    train_set = EEGDataset(root_dir=dataset.root_dir, 
                           data=dataset.x[:idx_sep,:,:],
                           labels=dataset.y[:idx_sep],
                           info=dataset.info[:idx_sep])
    test_set = EEGDataset(root_dir=dataset.root_dir,
                          data=dataset.x[idx_sep:,:,:],
                          labels=dataset.y[idx_sep:],
                          info=dataset.info[idx_sep:])
    
    return train_set, test_set
    
    

def my_train_test_split(dataset:EEGDatasetAdvanced, size=[], control_subject=False):
    
    '''Splits the EEGDatasetAdvanced into train and test set
    
    Arguments:
    dataset -- EEGDatasetAdvanced instance
    size -- a list len of 2 [train_size, test_size]
    control_subject -- wether need to manage subjects when splitting
                       the dataset (for a mult-subject dataset e.g.)
    '''
    
    assert len(size)==2, 'size must be given as [train_size, test_size] in parts [0.8, 0.2] e.g.'
    train_size, test_size = size
    
    train_set = deepcopy(dataset)
    test_set = deepcopy(dataset)    
    
    if control_subject:
        subjects = dataset.subjects.copy() if dataset.subjects != [] \
                   else dataset.available_subjects.copy()
        shuffle(subjects)
        idx_sep = round(len(subjects)*train_size)
        train_subjects = subjects[:idx_sep]
        test_subjects = subjects[idx_sep:]
        
        train_set.pick_subjects(train_subjects)
        test_set.pick_subjects(test_subjects)
        
        return train_set, test_set
    
    else:
        idx_sep = round(len(dataset)*train_size)
        
        train_set.data = dataset.data[:idx_sep]
        test_set.data = test_set.data[idx_sep:]
        train_set.info = dataset.info[:idx_sep]
        test_set.info = test_set.info[idx_sep:]
        
        return train_set, test_set
    
from copy import deepcopy

class One_vs_all():
    
    def __init__(self, dataset:EEGDatasetAdvanced):
        
        self.dataset = dataset
        self.subjects = dataset.subjects
        self.index = 0
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        if self.index >= len(self.subjects):
            raise StopIteration
        
        subjects = self.subjects.copy()
        test_subject = subjects.pop(self.index)
        train_set = deepcopy(self.dataset)
        test_set = deepcopy(self.dataset)
        train_set.pick_subjects(subjects)
        test_set.pick_subjects([test_subject])
        self.index+=1
        
        return train_set, test_set

from collections import Counter
    
def sampling(dataset:EEGDataset=None, X:np.ndarray=None, y:np.ndarray=None,
             report=False, mode:list=None, random_state=42):
    '''
    
    mode -- 'real', 'over', 'under', 'balanced'
    '''
    
    if dataset:
        X = flatten(dataset.x.copy()) if len(dataset.x.shape)>2 else dataset.x.copy() # to flatten channels dim
        y = dataset.y.copy()
    else:
        X = flatten(X.copy()) if len(X.shape)>2 else X.copy()
        y = y.copy()
    count = Counter(y)
    if report:
        print(f'x shape: {X.shape}\ny shape: {y.shape}')
        print(f'class ratio: target={count[0]}, non-target={count[1]}')
    
    mode = mode if mode else ['real','over', 'under', 'balanced']
    
    data = dict.fromkeys(mode)
    
    if 'real' in mode:
        data['real']={'x':X, 'y':y}
    
    # Make downsampling
    if 'under' in mode:
        n_target = count[0]
        y_down = np.hstack([y[y==0], y[y==1][:n_target]])
        X_down = np.vstack([X[y==0], X[y==1][:n_target]])
        data['under'] = {'x':X_down, 'y':y_down}
        count = Counter(y_down)
        if report:
            print(f'x downsampled shape: {X_down.shape}\ny downsampled shape: {y_down.shape}')
            print(f'class ratio (downsampled): target={count[0]}, non-target={count[1]}')
    
    # Make oversampling
    if 'over' in mode:
        oversamler = SMOTE(random_state=random_state)
        X_over, y_over = oversamler.fit_resample(X, y)
        data['over'] = {'x':X_over, 'y':y_over}
        count = Counter(y_over)
        if report:
            print(f'x oversampled shape: {X_over.shape}\ny oversampled shape: {y_over.shape}')
            print(f'class ratio (oversampled): target={count[0]}, non-target={count[1]}')
    
    # Oversampling of minor class and undersampling of major class
    if 'balanced' in mode:
        over = SMOTE(sampling_strategy=0.5, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
        pipe = ImbPipeline(steps=[('over', over), ('under', under)])
        X_balanced, y_balanced = pipe.fit_resample(X,y)
        data['balanced'] = {'x':X_balanced, 'y':y_balanced}
        count = Counter(y_balanced)
        if report:
            print(f'x balanced shape: {X_balanced.shape}\ny balanced shape: {y_balanced.shape}')
            print(f'class ratio (balanced): target={count[0]}, non-target={count[1]}')
    
    return data