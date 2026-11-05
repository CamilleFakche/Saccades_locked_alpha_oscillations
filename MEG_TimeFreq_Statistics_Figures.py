###############################################################################
##       MEG DATA ANALYSIS: POWER AND ITPC STATISTICS AND FIGURES            ##
###############################################################################
# by Camille Fakche 28/07/26

import mne 
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test
import numpy as np
import os 

dirMEGData = 'myMEGdatapath'
dirScript = 'myscriptpath'
os.chdir(dirScript)

ListSuj = ['B57D', 'B3EB', 'B5EB','B60D','B4E9','B5B3','B51C','B44F','B51F','B57B','B453','B457','B515','B3EE','B3F8','B4B8','B4B9','B4BB','B4BE', 'B4C1', 'B4C4', 'B4C6', 'B4C8','B57A','B57C','B58A', 'B329','B580','B581','B585','B588', 'B590', 'B3F3', 'B5E8', 'B591', 'B85B']
nsuj = len(ListSuj)

#%% Settings

ListCond = ['Correct','Incorrect','Diff']
nCond = len(ListCond)

ListData = ['Power','ITC']
nData = len(ListData)

ListSensors = ['mag','grad']
nSensors = len(ListSensors)

# Load Freqs
freqs_Filename = dirMEGData+'freqs.npy'
freqs = np.load(freqs_Filename, allow_pickle=True)
nFreqs = len(freqs)

# Load Times
times_Filename = dirMEGData+'times.npy'
times = np.load(times_Filename, allow_pickle=True)
nTimes = len(times)

# Load Chans List
chans_Filename = dirMEGData+'chans.npy'
chans = np.load(chans_Filename, allow_pickle=True)
nChans = len(chans)

# Define n Permutations
nperms = 1000

# Define p accepted
p_accept = 0.05

# Define tail
tail = 0
# If tail is 1, the statistic is thresholded above threshold. 
# If tail is -1, the statistic is thresholded below threshold. 
# If tail is 0, the statistic is thresholded on both sides of the distribution.

#%% Step0: Gather data across participants for permutations statistics 

for cond in range(nCond):
    for idata in range(nData):
        print(ListCond[cond])
        # Init Data
        Data_array = np.zeros((nsuj, nChans, nFreqs, nTimes))
        
        for suj in range(nsuj):
            print(ListSuj[suj])
            # Load data
            if ListCond[cond] == 'Diff':
                # Load Correct data
                DataCorrectFileSuj = dirMEGData+ListSuj[suj]+'/'+ListSuj[suj]+'_'+ListData[idata]+'_Correct-tfr.h5'
                DataCorrectSuj = mne.time_frequency.read_tfrs(DataCorrectFileSuj)
                DataCorrectSuj = DataCorrectSuj
                # Load Incorrect data
                DataIncorrectFileSuj = dirMEGData+ListSuj[suj]+'/'+ListSuj[suj]+'_'+ListData[idata]+'_Incorrect-tfr.h5'
                DataIncorrectSuj = mne.time_frequency.read_tfrs(DataIncorrectFileSuj)
                DataIncorrectSuj = DataIncorrectSuj
                # Compute the difference
                DataSuj = DataCorrectSuj - DataIncorrectSuj
            else:
                # Load data
                DataFileSuj = dirMEGData+ListSuj[suj]+'/'+ListSuj[suj]+'_'+ListData[idata]+'_'+ListCond[cond]+'-tfr.h5'
                DataSuj = mne.time_frequency.read_tfrs(DataFileSuj)
                DataSuj = DataSuj
            # Indent 
            DataSuj_array = DataSuj.data
            Data_array[suj,:,:,:] = DataSuj.data
            
        # Save Data array
        Data_arrayFile = dirMEGData+'AllSuj/AllSuj_'+ListData[idata]+'_'+ListCond[cond]+'.npy' 
        np.save(Data_arrayFile, Data_array)

#%% Step0 bis: Average data across participants for figures

for cond in range(nCond):
    for idata in range(nData):
        for suj in range(nsuj):
            print(ListSuj[suj])
            
            # Load data
            DataFileSuj = dirMEGData+ListSuj[suj]+'/'+ListSuj[suj]+'_'+ListData[idata]+'_'+ListCond[cond]+'-tfr.h5'
            DataSuj = mne.time_frequency.read_tfrs(DataFileSuj)
            
            # Sum across participants
            if suj == 0:
                Data = DataSuj
            else:
                Data += DataSuj
                
        # Average across participants
        Data /= nsuj
        
        # Save
        DataFile = dirMEGData+'AllSuj/AvgSuj_'+ListData[idata]+'_'+ListCond[cond]+'-tfr.h5'
        Data.save(DataFile, overwrite=True)

#%% Compute Adjacency Once - Same for every participants, every conditions

# Load Epoch data 
EpochFile = dirMEGData+'B3EB/B3EB_Epochs_SaccadeOnset_Correct.fif'
Epoch = mne.read_epochs(EpochFile, preload=False, verbose=True)
# Compute adjacency
adjacency_mag, ch_names_mag = find_ch_adjacency(Epoch.info, ch_type="mag")
adjacency_grad, ch_names_grad = find_ch_adjacency(Epoch.info, ch_type="grad")

# Plot adjacency
# mne.viz.plot_ch_adjacency(Epoch.info, adjacency_mag, ch_names_mag)
# mne.viz.plot_ch_adjacency(Epoch.info, adjacency_grad, ch_names_grad)

# Find mag and grad index
nMag = len(ch_names_mag)
nGrad = len(ch_names_grad)
index_mag = np.zeros(nMag, dtype=int)
index_grad = np.zeros(nGrad, dtype=int)
counter_mag = 0
counter_grad = 0
for ichan in range(nChans): 
    for imag in range(nMag):
        if ch_names_mag[imag] == chans[ichan]:
            index_mag[counter_mag] = int(ichan)
            counter_mag += 1
    for igrad in range(nGrad):
        if ch_names_grad[igrad] == chans[ichan]:
            index_grad[counter_grad] = int(ichan)
            counter_grad += 1   

#%% Step1: Permutations Statistsics

# Select Times of interest
tmin = -0.2
tmax = 0
index_tstart = np.array(np.where(times==tmin))
index_tend = np.array(np.where(times==max))
index_tstart = index_tstart[0,0]
index_tend = index_tend[0,0]
timesOI = times[index_tstart:index_tend]
nTimesOI = len(timesOI)

# Select Freqs of interest
fmin = 8
fmax = 13
index_fstart = np.array(np.where(freqs==fmin))
index_fend = np.array(np.where(freqs==fmax))
index_fstart = index_fstart[0,0]
index_fend = index_fend[0,0]
freqsOI = freqs[index_fstart:index_fend]
nFreqsOI = len(freqsOI)

for cond in range(nCond):
    for idata in range(nData):
        
        # Load Data
        DataFile = dirMEGData+'AllSuj/AllSuj_'+ListData[idata]+'_'+ListCond[cond]+'.npy'
        Data = np.load(DataFile, allow_pickle=True)
        
        # Inputs 
        # Raw Data [Suj * Chans * Freqs * Times]
        X_raw = Data
        # Transpose 
        # Data [Suj * Times * Freqs * Chans]
        X_trans = np.transpose(X_raw, (0, 3, 2, 1))
        # Select time-frequency window of interest
        X_OI = X_trans[:,index_tstart:index_tend,index_fstart:index_fend,:]
        # Average across times and frequencies
        X_OI_avgt = np.mean(X_OI, axis=1)
        X_OI_avgtf = np.mean(X_OI_avgt, axis=1)
        
        for isens in range(nSensors):
        
            # Sensor Selection
            if ListSensors[isens] == 'mag':
                X = X_OI_avgtf[:,index_mag]
                adjacency = adjacency_mag
            elif ListSensors[isens] == 'grad':
                X = X_OI_avgtf[:,index_grad]
                adjacency = adjacency_grad
            
            # Run Cluster Permutations Statistics
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                X,
                n_permutations=nperms,
                tail=tail,
                adjacency=adjacency,
                out_type="mask",
                verbose=True,
                )
            
            # Significant cluster
            nClusters = len(cluster_p_values)
            for iclus in range(nClusters):
                if cluster_p_values[iclus] < p_accept:
                    # Identify significant sensors
                    index_cluster = clusters[iclus]
                    # Save
                    index_cluster_Filename = dirMEGData+'AllSuj/AllSuj_'+ListData[idata]+'_'+ListCond[cond]+'_'+ListSensors[isens]+'_Sens_clusters'+str(iclus)+'.npy'
                    np.save(index_cluster_Filename, index_cluster)
                    
#%% Step2: Cluster Visualization 

# Index significant cluster
iclus=0

# Gradiometers Pairs Rule
# When only one gradiometer of the pair is significant
# 1: Keep the pair
# 0: Reject the pair
GradPairRule = 0 

for idata in range(nData):  
    for cond in range(nCond):
        # Load Data 
        DataFile = dirMEGData+'AllSuj/AvgSuj_'+ListData[idata]+'_'+ListCond[cond]+'-tfr.h5'
        Data = mne.time_frequency.read_tfrs(DataFile)
        
        for isens in range(nSensors):
            # Load Cluster
            cluster_Filename = dirMEGData+'AllSuj/AllSuj_'+ListData[idata]+'_'+ListCond[cond]+'_'+ListSensors[isens]+'_Sens_clusters'+str(iclus)+'.npy'
            cluster = np.load(cluster_Filename)
            
            # Sensors selection
            sensors = mne.pick_types(Data.info, meg=ListSensors[isens])
            sensors_cluster = sensors[cluster]
            Chan_List = Data.ch_names
            nChans = len(Chan_List)
            Chan_List_notSigSensors = Chan_List.copy()
            for ichan in range(nChans):
                if np.isin(ichan, sensors_cluster):
                    Chan_List_notSigSensors.remove(Chan_List[ichan])
            Data_SigSensors = Data.copy().drop_channels(Chan_List_notSigSensors)  
            
            # Plot Time-Frequency Map
            Data_SigSensors.plot(combine='mean', tmin=-0.4, tmax=0.4, fmin=4,vlim=[0, 0.015], cmap='RdBu_r')
            
            if ListSensors[isens] == 'grad':
                # MNE plot the pairs of gradiometers in topography
                # Combine the significance of each pair of gradiometers
                nGradPairs = int(len(cluster)/2)
                indexSigGradPairs = np.zeros(nGradPairs)
                for ipair in range(nGradPairs):
                    grad1 = ipair*2
                    grad2 = grad1+1
                    # Both gradiometers are significant 
                    if cluster[grad1]==True and cluster[grad2]==True:
                        indexSigGradPairs[ipair]=1
                    # Both gradiometers are non-significant
                    elif cluster[grad1]==False and cluster[grad2]==False:
                        indexSigGradPairs[ipair]=0
                    # Only one gradiometer of the pair is significant
                    elif cluster[grad1] != cluster[grad2]:
                        # Rule 1: Keep the pair
                        if GradPairRule == 1: 
                            indexSigGradPairs[ipair]=1
                        # Rule 0: Reject the pair
                        elif GradPairRule == 0:
                            indexSigGradPairs[ipair]=0
                # Turn into Boolean 
                cluster = np.array(indexSigGradPairs, dtype='bool') 
            
            # Topoplot with mask 
            Data.plot_topomap(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, vlim=[0, 0.03], ch_type=ListSensors[isens],
                              cmap='RdBu_r', mask=cluster, mask_params=dict(marker='*', markerfacecolor='k', 
                               markeredgecolor='k',linewidth=0, markersize=6))
            
            
            
            
            
                    

