###############################################################################
##                    MEG SOURCE ANALYSIS: FORWARD MODEL                     ##
###############################################################################
# by Camille Fakche 28/07/26

import mne
import os

dirMEGData = 'myMEGdatapath'
dirSourceData = 'mySourcedatapath'
dirFreeSurferData = 'myFreeSurferpath'
dirScript = 'myscriptpath'
os.chdir(dirScript)

# Example One Participant
suj = 'B5B3'

#%% Step0: FreeSurfer Reconstruction 

# Bash Script on Terminal -----------------------------------------------------

#!/bin/bash
# Define the SUBJECTS_DIR where recon-all will save output
# export SUBJECTS_DIR=dirFreeSurferData

# Define the FREESURFER LICENSE DIR 
# export FS_LICENSE=dirFreeSurferData/freesurfer_license.txt  

# Define the SUBJECT_ID
# subject_id=B5B3

# Deine the SUBJECT_INPUT
# input_file="${subject_id}/t1.nii.gz"

# Run recon-all 
# recon-all -i "$input_file" -s "$subject_id" -all 

# Run MNE-Python scalp surface and watershed BEM steps
# mne watershed_bem --overwrite --subject "$subject_id"

# Check FreeSurfer Reconstruction
Brain = mne.viz.get_brain_class()
brain = Brain(suj, 
              hemi='lh', 
              surf='pial',
              subjects_dir=dirFreeSurferData, 
              size=(800, 600))
brain.add_annotation('aparc.a2009s', borders=False)

#%% Step1: Create BEM Model

# Make BEM model
model = mne.make_bem_model(suj,
                          ico=4,
                          conductivity=(0.3,),
                          subjects_dir=dirFreeSurferData)
# Compute BEM
bem = mne.make_bem_solution(model)
# Save
mne.write_bem_solution(dirSourceData+suj+'_bem.fif', bem, overwrite=True)
# Plot 
mne.viz.plot_bem(subject=suj,
                subjects_dir=dirFreeSurferData,
                brain_surfaces='white',
                orientation='coronal')
    
#%% Step2: Coregistration with anatomical landmarks

# Manually
# See MNE tutorial here: https://www.youtube.com/watch?v=ALV5qqMHLlQ 
mne.gui.coregistration(subject=suj, subjects_dir=dirFreeSurferData)

# Check 
DataFile = dirMEGData+suj+'/'+suj+'.fif'
info = mne.io.read_raw(DataFile, verbose=False).info
TransformFile = dirSourceData+suj+'-trans.fif'
mne.viz.plot_alignment(info, trans=TransformFile, subject=suj, dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=dirFreeSurferData)


#%% Step3: Sources construction according to the BEM model

# Compute source 
surface = dirFreeSurferData+suj+'/'+'bem/inner_skull.surf'
source = mne.setup_volume_source_space(suj, subjects_dir=dirFreeSurferData,
                                     surface=surface,
                                     pos=5.0,
                                     verbose=True)
# Save
mne.write_source_spaces(dirSourceData+suj+'_sources.fif', source, overwrite=True)
# Plot
mne.viz.plot_bem(subject=suj, 
                 subjects_dir=dirFreeSurferData,
                 brain_surfaces='white', 
                 src=source, 
                 orientation='coronal')

#%% Step4: Construct Forward Model

# Load MEG info
DataFile = dirMEGData+suj+'/'+suj+'.fif'
info = mne.io.read_raw(DataFile, verbose=False).info

# Define trans filename
trans_filename = dirSourceData+suj+'-trans.fif'

# Define bem filename
bem_filename = dirSourceData+suj+'_bem.fif'

# Define source filename
source_filename = dirSourceData+suj+'_sources.fif'

# Compute forward model
fwd = mne.make_forward_solution(info, 
                            trans=trans_filename,
                            src=source_filename, 
                            bem=bem_filename,
                            meg=True, 
                            eeg=False, 
                            mindist=5,
                            n_jobs=-1, 
                            verbose=True)
# Save
fwd_filename = dirSourceData+suj+'_fwd.fif'
mne.write_forward_solution(fwd_filename, fwd, overwrite=True)

    