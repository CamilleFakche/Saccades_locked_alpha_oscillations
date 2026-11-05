###############################################################################
##            MEG SOURCE ANALYSIS: LCMV STATISTICS AND FIGURES               ##
###############################################################################
# by Camille Fakche 28/07/26

import nilearn.plotting
import nilearn.image
import nibabel as nib
from nibabel.freesurfer.io import read_annot
import mne
import numpy as np
from scipy import sparse
import os 
import pandas as pd
import pickle

dirSourceData = 'mySourcedatapath'
dirFreeSurferData = 'myFreeSurferpath'
dirScript = 'myscriptpath'
os.chdir(dirScript)

ListSuj = ['B57D', 'B3EB', 'B5EB','B60D','B4E9','B5B3','B51C','B44F','B51F','B57B','B453','B457','B515','B3EE','B3F8','B4B8','B4B9','B4BB','B4BE', 'B4C1', 'B4C4', 'B4C6', 'B4C8','B57A','B57C','B58A', 'B329','B580','B581','B585','B588', 'B590', 'B3F3', 'B5E8', 'B591', 'B85B']
nsuj = len(ListSuj)

idata = 'ITPC'
cond = '_Diff_Saccades_CorrectmIncorrect'

ListSensor = ['grad','mag']
nSensor = len(ListSensor)

#%%

def load_object(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    return obj

#%% Step0: Load FEF Localizer from Bedini et al., 2023

# Load FEF Localizer (Nifti image)
fef_file = dirSourceData+'FEF_localizer_sample_MNI_all_FWE01_5k_ALE.nii'
fef = nib.load(fef_file)

#%% Step0: Glasser Atlas 

# Read Labels
# Left
labels, ctab, names = read_annot(r'U:/Glasser/lh.HCP-MMP1.annot')
left = {i: names[i].decode() for i in range(1, len(names))}
# Right
labels, ctab, names = read_annot(r'U:/Glasser/rh.HCP-MMP1.annot')
right = {i + 180: names[i].decode() for i in range(1, len(names))}
glasser_labels = {**left, **right}

# Read atlas
glasser_img = nib.load(dirSourceData+'glasser_MNI152NLin6Asym_labels_p20.nii')

#%% Step1: Combine participant throught NIFTII conversion

for sensor in range(nSensor):
    # Init
    img_allSuj = []
            
    for suj in range(nsuj):
                
        # Load Forward Model
        fwd_filename = dirSourceData+suj+'_fwd.fif'
        fwd = mne.read_forward_solution(fwd_filename)
        src = fwd['src']
                
        # Load Vol Source Estimate File
        stc_filename =  dirSourceData+suj+'_LCMVREV_'+ListSensor[sensor]+'_'+idata+cond
        stc = load_object(stc_filename)
                
        # Convert to NIFTI
        img = stc.as_volume(src) 
        del fwd
        del src
        del stc
                
        # Get data 
        data = img.get_fdata() 
        # Mean across times
        mean_data = np.mean(data,axis=3)
        # Recreate NIFTI image
        affine = img.affine  
        header = img.header 
        mean_img = nib.Nifti1Image(mean_data, affine, header)
                
        # Sample NIFTI to fef
        mean_img_sample = nilearn.image.resample_to_img(
            mean_img,     
            fef,      
            interpolation="continuous")
                
        # Indent
        img_allSuj.append(mean_img_sample)
                
        # Combine
        img_allSuj_4D = nilearn.image.concat_imgs(img_allSuj)
            
        # Save
        img_allSuj_4D_file = dirSourceData+'allSuj_LCMV_'+ListSensor[sensor]+'_'+idata+cond+'.nii'
        nib.save(img_allSuj_4D, img_allSuj_4D_file)

#%% Step2: Cluster-based permutations 

FEF = 1 # Perform statistics on FEF 

for sensor in range(nSensor):
    
    # Load NIFTII 
    img_allSuj_4D_file = dirSourceData+'allSuj_LCMV_'+ListSensor[sensor]+'_'+idata+cond+'.nii'
    img_allSuj_4D = nib.load(img_allSuj_4D_file)
    # Get data 
    data_4D = img_allSuj_4D.get_fdata()
            
    if FEF:
        # FEF Mask
        fef_data = fef.get_fdata()
        mask = (fef_data > 0)[:, :, :, np.newaxis]
        data_4D = data_4D * mask
    
    # Dimensions
    nx, ny, nz, nsuj = data_4D.shape
    # Transform as nSujs x nVoxels 
    X = data_4D.reshape(nx*ny*nz, nsuj).T

    # Adjacency
    shape = (nx, ny, nz)
    idx = np.arange(np.prod(shape)).reshape(shape)
    rows = []
    cols = []
    # 6-connectivity neighbors
    for axis in range(3):
        for shift in [-1, 1]:
            shifted = np.roll(idx, shift, axis=axis)
            rows.extend(idx.ravel())
            cols.extend(shifted.ravel())
        adjacency = sparse.coo_matrix((np.ones(len(rows)),(rows, cols)),shape=(np.prod(shape), np.prod(shape))).tocsr()
            
    # Permutations
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=1000,
        tail=0,
        n_jobs=-1)
            
    # Significant cluster
    nClusters = len(cluster_p_values)
    for iclus in range(nClusters):
        if cluster_p_values[iclus] < 0.05:
            # Identify significant sensors
            index_sigCluster = clusters[iclus]
            # Save
            index_sigCluster_Filename = dirSourceData+'allSuj_LCMVREV_'+ListSensor[sensor]+'_'+idata+cond+'_clusters'+str(iclus)+'.npy'
            np.save(index_sigCluster_Filename, index_sigCluster)
            
            
#%% Step3: Plot cluster-based permutations

t1_fname = dirFreeSurferData+'fsaverage/mri/T1.mgz'

# Index significant cluster
iclus=0

for sensor in range(nSensor):
    
    # Load NIFTII 
    img_allSuj_4D_file = dirSourceData+'allSuj_LCMV_'+ListSensor[sensor]+'_'+idata+cond+'.nii'
    img_allSuj_4D = nib.load(img_allSuj_4D_file)
            
    # Load significant cluster
    indexSourcesFile = dirSourceData+'allSuj_LCMVREV_'+ListSensor[sensor]+'_'+idata+cond+'_clusters'+str(iclus)+'.npy'
    indexSources = np.load(indexSourcesFile)
      
    # Get data
    data_4D = img_allSuj_4D.get_fdata()
            
    # Average across participants 
    data = np.mean(data_4D, axis=3)
                
    # Transform to 1D voxels
    vecteur = data.flatten()
    # Identify significant sources 
    mask = np.zeros(vecteur.shape, dtype=bool)
    mask[indexSources] = True
    vecteur *= mask
    # Recreate 3D
    data_cluster = vecteur.reshape(data.shape)
    # Recreate NIFTII
    affine = img_allSuj_4D.affine  
    header = img_allSuj_4D.header 
    img_cluster = nib.Nifti1Image(data_cluster, affine, header)
            
    # Plot
    print(ListSensor[sensor]+'_'+idata+cond)
    min_val = np.min(data_cluster)
    max_val = np.max(data_cluster)
    nilearn.plotting.plot_stat_map(img_cluster, str(t1_fname), draw_cross=False, black_bg=True, vmin=min_val, vmax=max_val, 
                       cut_coords=(35, 0, 45))
            
            
#%% Step4: Labels

for sensor in range(nSensor):
    
    # Load NIFTII 
    img_allSuj_4D_file = dirSourceData+'allSuj_LCMV_'+ListSensor[sensor]+'_'+idata+cond+'.nii'
    img_allSuj_4D = nib.load(img_allSuj_4D_file)
            
    # Load significant cluster
    indexSourcesFile = dirSourceData+'allSuj_LCMVREV_'+ListSensor[sensor]+'_'+idata+cond+'_clusters'+str(iclus)+'.npy'
    indexSources = np.load(indexSourcesFile)
    
    # Get data
    data_4D = img_allSuj_4D.get_fdata()  
    
    # Average across participants 
    data = np.mean(data_4D, axis=3)  
    
    # Transform to 1D voxels
    vecteur = data.flatten()
    # Identify significant sources 
    mask = np.zeros(vecteur.shape, dtype=bool)
    mask[indexSources] = True
    vecteur *= mask
    # Recreate 3D
    data_cluster = vecteur.reshape(data.shape)

    # Resample Glasser atlas to my data 
    from nilearn.image import resample_to_img
    glasser_2mm = resample_to_img(
        glasser_img,
        img_cluster,
        interpolation="nearest"
        )
    glasser_data = glasser_2mm.get_fdata().astype(int)
        
    # Identify labels of positive sources
    positive_mask = data_cluster > 0
    results_positive = []
    positive_roi_ids = glasser_data[positive_mask]
    ids, counts = np.unique(positive_roi_ids, return_counts=True)
    for roi_id, count in zip(ids, counts):
        if roi_id == 0:
            continue
        roi_mask = (positive_mask & (glasser_data == roi_id))
        values = data_cluster[roi_mask]
        results_positive.append({
            "ROI_ID": roi_id,
            "ROI_name": glasser_labels[roi_id],
            "n_voxels_positive": len(values),
            "positive_activity_sum": np.sum(values),
            "mean_positive_activity": np.mean(values)})
        
df_positive = pd.DataFrame(results_positive)
df_positive_sorted = df_positive.sort_values("positive_activity_sum", ascending=False)
with pd.option_context('display.max_rows', None):
    print(df_positive_sorted)

            
