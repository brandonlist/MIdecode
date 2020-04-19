import torch
from torch.utils.data import dataset
import numpy as np

# class EEG_ML_database(dataset):
#     """
#     Creating Machine learning datasets of EEG data
#     """
#     def __init__(self):
#         super(EEG_ML_database, self).__init__()
#
#     def __len__(self):
#         pass
#
#     def __getitem__(self, idx):
#         pass


class Trial(object):
    """
    Totally abstract object
    attributes can be added as needed
    We recommend the following attribute names to be identical when reading different datasets:
        signal
        target
        sample_rate...
    """
    pass


class EEG_data_subject(object):
    """
    EEG data for one subject
    expected updating to succeed self-define general form of data
    """
    def __init__(self,preload,id):
        self.preload = preload
        self.id = id

    def load_data(self,*args,**kwargs):
        "parasing different form of data"
        self.n_trial = kwargs['n_trial']
        self.subject_trials = []
        for i in range(self.n_trial):
            trial = Trial()
            self.subject_trials.append(trial)


class EEG_database(object):
    """
    General EEG data baseform, designed for multiple general-purpose use of the data, including:
    1.creat machine learning database
    2.analysis of the data including visualization
    3....
    Things like sample frequency are not included in the baseform.
    They can be added to the baseform or the Trial object as needed.
    """
    def __init__(self,name,n_subject,preload=True):
        self.preload = preload
        self.n_subject = n_subject
        self.name = name

    def load_data(self,*args):
        print("loading datasets: {0} ".format(self.name))
        self.subjects_data = []
        for i in range(self.n_subject):
            subject = EEG_data_subject(preload=True,id=i)
            self.subjects_data.append(subject)


    def create_ml_datasets(self):
        pass

    pass
