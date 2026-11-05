# Alpha Oscillations Shape Saccade Timing
The codes provided here allow to perform the main analyses from Fakche C., and Jensen O., (2026), Temporal Coordination of Saccades by Alpha Oscillations in an extended network including the Frontal Eye Field, with Python 3.12.9. and MNE 1.10.
## Raw Data 
https://doi.org/10.25500/edata.bham.00001170 
## Codes
### MEG_TimeFreq.py
Multitapers time-frequency decomposition on single epochs with subsampling procedure. 
### MEG_TimeFreq_Statistics_Figures.py
Cluster-based permutations at the group level for power and ITPC, constrained to alpha frequencies and time window before saccade onset. Topographies and time-frequency representations of the significant cluster. 
### MEG_ForwardModel.py
Computation of a volumetric forward model using a 5-mm source grid, including FreeSurfer reconstruction, BEM model generation, MEG–MRI coregistration based on anatomical landmarks, source space construction, and forward solution estimation.
### MEG_LCMV.py
LCMV beamforming pipeline for MEG source reconstruction and extraction of oscillatory amplitude and phase-locking measures, including signal preprocessing, covariance matrix estimation, spatial filter computation, Hilbert transform, source projection, and extraction of amplitude and inter-trial phase coherence (ITPC).
### MEG_LCMV_Statistics_Figures.py
MEG LCMV group-level analysis pipeline including permutation statistics, visualization, and Glasser atlas-based labeling with optional FEF source constraint.
