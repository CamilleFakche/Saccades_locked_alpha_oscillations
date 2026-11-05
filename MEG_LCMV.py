###############################################################################
##                    MEG SOURCE ANALYSIS: LCMV                              ##
###############################################################################
# by Camille Fakche 28/07/26

import mne 
import os
import numpy as np
import pickle

dirMEGData = 'myMEGdatapath'
dirSourceData = 'mySourcedatapath'
dirScript = 'myscriptpath'
os.chdir(dirScript)

# Example One Participant
suj = 'B5B3'

fmin = 8
fmax = 13

# Define conditions 
cond_name_1 = 'Correct'
cond_name_2 = 'Incorrect'
sensor = 'mag'

#%%

def save_object(obj, filename): 
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

#%% Step0: Load Data

# Load froward model
fwd_filename = dirSourceData+suj+'_fwd.fif'
fwd = mne.read_forward_solution(fwd_filename)

# Load Epoch data
EpochFileAll = dirMEGData+suj+'/'+suj+'_Epochs_SaccadeOnset_All.fif'
epochAll = mne.read_epochs(EpochFileAll, preload=True, verbose=True)

# Load Epoch data
Epoch1File = dirMEGData+suj+'/'+suj+'_Epochs_SaccadeOnset_'+cond_name_1+'.fif'
epoch1 = mne.read_epochs(Epoch1File, preload=True, verbose=True)

Epoch2File = dirMEGData+suj+'/'+suj+'_Epochs_SaccadeOnset_'+cond_name_2+'.fif'
epoch2 = mne.read_epochs(Epoch2File, preload=True, verbose=True)

# Load Sensors Names
ch_names_arr_filename = dirSourceData+'ch_names_'+sensor+'.npy'
ch_names_arr = np.load(ch_names_arr_filename)
ch_names = ch_names_arr.tolist()

#%% Step1: Downsample, Filter, Crop

# Select Sensors
epochAll.pick_channels(ch_names)
epoch1.pick_channels(ch_names)
epoch2.pick_channels(ch_names)

# Downsample
epochAll.resample(200)
epoch1.resample(200)
epoch2.resample(200)

# Filter
epochAll.filter(fmin, fmax)
epoch1.filter(fmin, fmax)
epoch2.filter(fmin, fmax)

# Crop
epochAll.crop(tmin=-0.4, tmax=0.4)
epoch1.crop(tmin=-0.2, tmax=0)
epoch2.crop(tmin=-0.2, tmax=0)

#%% Step2: Compute Covariance Matrix

# Compute Covariance matrix
DataAllCov = mne.compute_covariance(epochAll, method="empirical")

# Compute the rank 
rank = mne.compute_rank(epochAll, tol=1e-6, tol_kind='relative', proj=False)
print(rank)

#%% Step3: Compute Spatial Filters

filters = mne.beamformer.make_lcmv(epochAll.info, fwd, DataAllCov, reg=0.05, pick_ori="max-power", weight_norm="unit-noise-gain", rank=rank, reduce_rank=True)

#%% Step4: Hilbert Transform

epochAllHilb = epochAll.apply_hilbert()
epoch1Hilb = epoch1.apply_hilbert()
epoch2Hilb = epoch2.apply_hilbert()

#%% Step5: Apply Spatial Filters

srcsAll = mne.beamformer.apply_lcmv_epochs(epochAllHilb, filters)
srcs1 = mne.beamformer.apply_lcmv_epochs(epoch1Hilb, filters)
srcs2 = mne.beamformer.apply_lcmv_epochs(epoch2Hilb, filters)

#%% Intermediate Step: Subsampling

# Subsampling 
nTrials_1 = len(srcs1)
nTrials_2 = len(srcs2)
if nTrials_1 > nTrials_2:
    nTrials = nTrials_2
    nTrials_2_sub = np.arange(0,nTrials_2,1)
    # Select nTrials_2 trials from srcs1
    nTrials_1_sub = np.random.choice(np.arange(0,nTrials_1,1),nTrials_2)
elif nTrials_2 > nTrials_1:
    nTrials = nTrials_1
    nTrials_1_sub = np.arange(0,nTrials_1,1)
    # Select nTrials_1 trials from srcs2
    nTrials_2_sub = np.random.choice(np.arange(0,nTrials_2,1),nTrials_1)
    
#%% Step6: Extract the Amplitude 

# srcsCorr is a list of nTrials = nVolSourceEstimate
# with VolSourceEstimate.data = (nsources * ntimes)

# srcs All    
nTrialsAll = len(srcsAll)
for trial in range(nTrialsAll):
    # Extract Data Trial
    DataTrial = srcsAll[trial].data
    # Init
    if trial == 0:
        nSources = DataTrial.shape[0]
        nTimes =  DataTrial.shape[1]
        DataTrialAmp = np.zeros((nTrialsAll, nSources, nTimes))
    # Amplitude Enveloppe
    DataTrialAmp[trial,:,:] = np.abs(DataTrial)
# Mean
DataTrialAmp = np.squeeze(np.mean(DataTrialAmp, axis=0))

# Format to VolSourceEstimate 
# Mean
srcsAllAmpMean = srcsAll[0].copy()
srcsAllAmpMean.data = DataTrialAmp

# Save
srcsAllAmpMean_filename =  dirSourceData+suj+'_LCMV_'+sensor+'_Amplitude_Saccades_All'
save_object(srcsAllAmpMean, srcsAllAmpMean_filename)


#%% Step7: Extract ITPC 

# srcsCorr is a list of nTrials = nVolSourceEstimate
# with VolSourceEstimate.data = (nsources * ntimes)

# srcs 1
for trial_index in range(nTrials):
    trial = nTrials_1_sub[trial_index]
    # Extract Data Trial
    DataTrial = srcs1[trial].data
    # Init
    if trial_index == 0:
        nSources = DataTrial.shape[0]
        nTimes =  DataTrial.shape[1]
        DataAllTrial = np.zeros((nTrials, nSources, nTimes), dtype=complex)
    DataAllTrial[trial_index,:,:,] = DataTrial # Complex
ITPC1 = np.squeeze(np.abs(np.mean(DataAllTrial/np.abs(DataAllTrial), axis=0)))
del DataAllTrial

# srcs 2
for trial_index in range(nTrials):
    trial = nTrials_2_sub[trial_index]
    # Extract Data Trial
    DataTrial = srcs2[trial].data
    # Init
    if trial_index == 0:
        nSources = DataTrial.shape[0]
        nTimes =  DataTrial.shape[1]
        DataAllTrial = np.zeros((nTrials, nSources, nTimes), dtype=complex)
    DataAllTrial[trial_index,:,:,] = DataTrial # Complex 
ITPC2 = np.squeeze(np.abs(np.mean(DataAllTrial/np.abs(DataAllTrial), axis=0)))
del DataAllTrial

# Format to VolSourceEstimate 
# Cond1
src1ITPC = srcs1[0].copy()
src1ITPC.data = ITPC1
# InCorr
src2ITPC = srcs2[0].copy()
src2ITPC.data = ITPC2
# Diff
srcDiffITPC = srcs1[0].copy()
srcDiffITPC.data = ITPC1 - ITPC2

# Save
src1ITPC_filename =  dirSourceData+suj+'_LCMV_'+sensor+'_ITPC_Saccades_'+cond_name_1
save_object(src1ITPC, src1ITPC_filename)
# Save
src2ITPC_filename =  dirSourceData+suj+'_LCMV_'+sensor+'_ITPC_Saccades_'+cond_name_2
save_object(src2ITPC, src2ITPC_filename)
# Save
srcDiffITPC_filename =  dirSourceData+suj+'_LCMVREV_'+sensor+'_ITPC_Diff_Saccades_'+cond_name_1+'m'+cond_name_2
save_object(srcDiffITPC, srcDiffITPC_filename)

