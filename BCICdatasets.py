from scipy.io import loadmat
import torch
from BCI_database import EEG_database
import numpy as np
import numpy.ma as npm
import os
import mne


class BCIC2_3(EEG_database):
    """
    datasets:BCI Competition II 3
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC2\3\dataset_BCIcomp1.mat'):
        super(BCIC2_3, self).__init__(name='BCIC23',n_subject=1)
        self.path = path

    def load_data(self,*args):
        data_all = loadmat(self.path)
        x_data = data_all['x_train']
        y_data = data_all['y_train']
        x_data = np.swapaxes(x_data, 0, 2)
        # x_data:[140(trials),3(channels),1152(time_steps)]
        super(BCIC2_3, self).load_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            self.subjects_data[i].load_data(n_trial=140)
            for idx,trial in enumerate(x_data):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = y_data[idx]


class BCIC3_3a(EEG_database):
    """
    datasets:BCI Competition III 3a
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\3\3a'):
        super(BCIC3_3a, self).__init__(name='BCIC33a',n_subject=3)
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun','785':'really unknown1','786':'really unknown2'}
        self.classes = ['cueOnsetLeft','cueOnsetRight','cueOnsetTongue','cueOnsetFoot']

    def load_data(self,*args):
        super(BCIC3_3a, self).load_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "

            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s,file_path)

            raw = mne.io.read_raw_gdf(path_s)
            event, _ = mne.events_from_annotations(raw)
            event_id = {}
            for code in _:
                event_id[self.eventDescription[code]] = _[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            epochs = mne.Epochs(raw, event, event_id, event_repeated='merge')
            #compute n_trail
            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    count = count+1
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = class_i
                    count = count+1


class BCIC3_3b(EEG_database):
    """
    datasets:BCI Competition III 3b
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\3\3b'):
        super(BCIC3_3b, self).__init__(name='BCIC33b',n_subject=3)
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun','785':'really unknown1','781':'really unknown2'}
        self.classes = ['cueOnsetLeft','cueOnsetRight']

    def load_data(self,*args):
        super(BCIC3_3b, self).load_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "

            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s,file_path)

            raw = mne.io.read_raw_gdf(path_s)
            event, _ = mne.events_from_annotations(raw)
            event_id = {}
            for code in _:
                event_id[self.eventDescription[code]] = _[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            epochs = mne.Epochs(raw, event, event_id, event_repeated='merge')
            #compute n_trail
            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    count = count+1
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = class_i
                    count = count+1


class BCIC3_4a(EEG_database):
    """
    datasets:BCI Competition III 4a
    see   for more information
    bug0: in load_data, the time length of a trial (length) can not be too big due to menmery failure
        in fact it can only be small numbers like 1 now (wkl) future adaption of preload-false
        method is severely needed
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\4\4a\1000Hz'):
        super(BCIC3_4a, self).__init__(name='BCIC34a',n_subject=5)
        self.path = path
        self.classes=['1','2']

    def load_data(self,*args,length = 1):
        super(BCIC3_4a, self).load_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path = [p for p in os.listdir(os.path.join(self.path,'s'+str(i+1))) if p.split('.')[-1] =='mat'][0]
            file_mat_path = os.path.join(self.path,'s'+str(i+1),file_mat_path)
            data_subject_i = loadmat(file_mat_path)
            mask_all = data_subject_i['mrk']['y'].item() < 5
            n_trial = mask_all.sum()
            count = 0
            self.subjects_data[i].load_data(n_trial=n_trial)
            for class_i in self.classes:
                count_i =0
                mask_i = data_subject_i['mrk']['y'].item() == int(class_i)
                n_trial_i = mask_i.sum()
                index_list = npm.array(data_subject_i['mrk']['pos'].item(), mask=mask_i)
                index_list = [ind for ind in index_list[0] if ind != False]
                for trial_i in range(n_trial_i):
                    signal = data_subject_i['cnt'][index_list[count_i]:index_list[count_i]+length,:]
                    signal = np.swapaxes(signal,0,1)
                    self.subjects_data[i].subject_trials[count].signal = signal
                    self.subjects_data[i].subject_trials[count].target = class_i
                    count = count+1
                    count_i = count_i+1


class BCIC3_4b(EEG_database):
    """
    datasets:BCI Competition III 4b
    see   for more information
    bug0: in load_data, the time length of a trial (length) can not be too big due to menmery failure
        in fact it can only be small numbers like 1 now (wkl) future adaption of preload-false
        method is severely needed
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\4\4b\1000Hz'):
        super(BCIC3_4b, self).__init__(name='BCIC34b',n_subject=1)
        self.path = path
        self.classes=['1','-1']

    def load_data(self,*args,length = 1):
        super(BCIC3_4b, self).load_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path_train = os.path.join(self.path,'data_set_IVb_al_train.mat')
            data_subject_i = loadmat(file_mat_path_train)
            n_trial = abs(data_subject_i['mrk']['y'].item()).sum()
            count = 0
            self.subjects_data[i].load_data(n_trial=n_trial)
            for class_i in self.classes:
                count_i =0
                mask_i = data_subject_i['mrk']['y'].item() == int(class_i)
                n_trial_i = mask_i.sum()
                index_list = npm.array(data_subject_i['mrk']['pos'].item(), mask=mask_i)
                index_list = [ind for ind in index_list[0] if ind != False]
                for trial_i in range(n_trial_i):
                    signal = data_subject_i['cnt'][index_list[count_i]:index_list[count_i]+length,:]
                    signal = np.swapaxes(signal,0,1)
                    self.subjects_data[i].subject_trials[count].signal = signal
                    self.subjects_data[i].subject_trials[count].target = class_i
                    count = count+1
                    count_i = count_i+1



def test_BCIC33a():
    t = BCIC3_3a()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    retur
def test_BCIC23():
    t = BCIC2_3()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    return
def test_BCIC33b():
    t = BCIC3_3b()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    return
def test_BCIC34a():
    t = BCIC3_4a()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    return
def test_BCIC34b():
    t = BCIC3_4b()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    return
def test_BCIC34b():
    t = BCIC3_4b()
    t.load_data()
    t.subjects_data[0].subject_trials[0].signal
    type(t)   #BCIC2_3
    type(t.subjects_data)  #list
    type(t.subjects_data[0])  #BCI_database.EEG_data_subject
    type(t.subjects_data[0].subject_trials) #list
    type(t.subjects_data[0].subject_trials[0]) #BCI_database.Trial
    type(t.subjects_data[0].subject_trials[0].signal) #numpy.ndarray
    type(t.subjects_data[0].subject_trials[0].target) #numpy.ndarray
    #iteration
    for s in t.subjects_data:
        for trial in s.subject_trials:
            print(trial.target)
    return
