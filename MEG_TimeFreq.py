###############################################################################
##               MEG DATA ANALYSIS: TIME FREQUENCY DECOMPOSITION             ##
###############################################################################
# by Camille Fakche 28/07/26

import mne
import numpy as np
import os 

dirMEGData = 'myMEGdatapath'
dirScript = 'myscriptpath'
os.chdir(dirScript)

# Example One Participant
suj = 'B3EB'

#%% Settings

freqs = np.arange(2, 31, 1)
n_cycles = freqs / 2.  
time_bandwidth = 2.0
nSubsampling = 50

# Define conditions 
cond_name_1 = 'Correct'
cond_name_2 = 'Incorrect'

#%% Time-frequency decomposition - Multitaper with Subsampling
  
# Load Epoch data
Epoch1File = dirMEGData+suj+'/'+suj+'_Epochs_SaccadeOnset_'+cond_name_1+'.fif'
epoch1 = mne.read_epochs(Epoch1File, preload=True, verbose=True)

Epoch2File =  dirMEGData+suj+'/'+suj+'_Epochs_SaccadeOnset_'+cond_name_2+'.fif'
epoch2 = mne.read_epochs(Epoch2File, preload=True, verbose=True)

# Define nTrials
nEpoch1 = len(epoch1)
nEpoch2 = len(epoch2)

# Subsampling procedure 
for sub in range(nSubsampling):
    if nEpoch1 > nEpoch2:
        epoch2sub = epoch2
        # Select nEpoch2 trials from epoch1
        trials_index = np.random.choice(np.arange(0,nEpoch1,1),nEpoch2)
        epoch1sub = epoch1.__getitem__(trials_index)
    elif nEpoch1 < nEpoch2:
        epoch1sub = epoch1
        # Select nEpoch1 trials from epoch2
        trials_index = np.random.choice(np.arange(0,nEpoch2,1),nEpoch1)
        epoch2sub = epoch2.__getitem__(trials_index)
    elif nEpoch1 == nEpoch2:
        epoch1sub = epoch1
        epoch2sub = epoch2
        
    # Time-frequency decomposition cond1
    power_mne_epoch1_sub, itc_mne_epoch1_sub = mne.time_frequency.tfr_multitaper(
        epoch1sub,
        freqs=freqs, 
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, 
        use_fft=True, 
        decim=2,
        n_jobs=4,
        verbose=True) 
    
    # Time-frequency decomposition cond2
    power_mne_epoch2_sub, itc_mne_epoch2_sub = mne.time_frequency.tfr_multitaper(
        epoch2sub,
        freqs=freqs, 
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, 
        use_fft=True, 
        decim=2,
        n_jobs=4,
        verbose=True) 
    
    # Sum across subsampling
    if sub == 0:
        power_mne_epoch1_sub_all = power_mne_epoch1_sub
        itc_mne_epoch1_sub_all = itc_mne_epoch1_sub
        power_mne_epoch2_sub_all = power_mne_epoch2_sub
        itc_mne_epoch2_sub_all = itc_mne_epoch2_sub
    else:
        power_mne_epoch1_sub_all += power_mne_epoch1_sub
        itc_mne_epoch1_sub_all += itc_mne_epoch1_sub
        power_mne_epoch2_sub_all += power_mne_epoch2_sub
        itc_mne_epoch2_sub_all += itc_mne_epoch2_sub
        
# Divide across subsampling
power_mne_epoch1_sub_all /= nSubsampling
power_mne_epoch2_sub_all /= nSubsampling
itc_mne_epoch1_sub_all /= nSubsampling
itc_mne_epoch2_sub_all /= nSubsampling
    
# Save 
PowerFile1 =  dirMEGData+suj+'/'+suj+'_Power_'+cond_name_1+'-tfr.h5'
power_mne_epoch1_sub_all.save(PowerFile1, overwrite=True)
    
ITCFile1 =  dirMEGData+suj+'/'+suj+'_ITC_'+cond_name_1+'-tfr.h5'
itc_mne_epoch1_sub_all.save(ITCFile1, overwrite=True)
    
PowerFile2 =  dirMEGData+suj+'/'+suj+'_Power_'+cond_name_2+'-tfr.h5'
power_mne_epoch2_sub_all.save(PowerFile2, overwrite=True)
    
ITCFile2 =  dirMEGData+suj+'/'+suj+'_ITC_'+cond_name_2+'-tfr.h5'
itc_mne_epoch2_sub_all.save(ITCFile2, overwrite=True)


#%% Save Freqs, Times, Chans once

freqs = power_mne_epoch1_sub_all.freqs
freqs_Filename = dirMEGData+'freqs.npy'
np.save(freqs_Filename, freqs)

times = power_mne_epoch1_sub_all.times
times_Filename = dirMEGData+'times.npy'
np.save(times_Filename,times)

# Save Chans List
chans = power_mne_epoch1_sub_all.ch_names
chans_Filename = dirMEGData+'chans.npy'
np.save(chans_Filename, chans)


