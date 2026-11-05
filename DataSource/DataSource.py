
import numpy as np
import mne 

#%% Functions - Eye Data 

import matplotlib.pyplot as plt

def scatter_plot(data, ylim):
    
    import matplotlib.colors as mcolors
    colors = mcolors.CSS4_COLORS
    color1 = colors['mediumturquoise']
    color2 = colors['crimson']
    
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    nsuj = len(data)
    x_axis = np.ones([nsuj])
    # nSaccades per trial
    axs[0].scatter(x_axis,data[:,0], 40, color=color1)
    axs[0].scatter(2*x_axis,data[:,1], 40, color=color2)
    axs[0].scatter(1, np.mean(data[:,0]), 80, color=color1)
    axs[0].scatter(2, np.mean(data[:,1]), 80, color=color2)
    axs[0].set_xticks([1, 2], labels=['Correct', 'Incorrect']) 
    axs[0].set_xlabel('Number Saccades per Trial')
    axs[0].set_ylim(ylim)
    # Saccade duration
    axs[1].scatter(x_axis,data[:,2]*1000, 40, color=color1)
    axs[1].scatter(2*x_axis,data[:,3]*1000, 40, color=color2)
    axs[1].scatter(1, np.mean(data[:,2]*1000), 80, color=color1)
    axs[1].scatter(2, np.mean(data[:,3]*1000), 80, color=color2)
    axs[1].set_xticks([1, 2], labels=['Correct', 'Incorrect']) 
    axs[1].set_xlabel('Saccade Duration (ms)')
    axs[1].set_ylim([20, 60])
    # Saccade amplitude
    axs[2].scatter(x_axis,data[:,4], 40, color=color1)
    axs[2].scatter(2*x_axis,data[:,5], 40, color=color2)
    axs[2].scatter(1, np.mean(data[:,4]), 80, color=color1)
    axs[2].scatter(2, np.mean(data[:,5]), 80, color=color2)
    axs[2].set_xticks([1, 2], labels=['Correct', 'Incorrect']) 
    axs[2].set_xlabel('Saccade Amplitude (°)')
    axs[2].set_ylim([3, 7])
    # Saccade velocity
    axs[3].scatter(x_axis,data[:,6], 40, color=color1)
    axs[3].scatter(2*x_axis,data[:,7], 40, color=color2)
    axs[3].scatter(1, np.mean(data[:,6]), 80, color=color1)
    axs[3].scatter(2, np.mean(data[:,7]), 80, color=color2)
    axs[3].set_xticks([1, 2], labels=['Correct', 'Incorrect']) 
    axs[3].set_xlabel('Saccade Velocity (°/s)')
    axs[3].set_ylim([200, 400])
    
def saccade_sequence(data):
    
    from scipy.optimize import curve_fit
    
    amplitude = data[0,:]
    velocity = data[1,:]
    fig, ax = plt.subplots()
    plt.hexbin(amplitude,velocity, gridsize=40, mincnt=1)
    plt.colorbar(label="nSaccades")
    ax.set_xlabel("Saccade magnitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")
    def main_sequence(x, a, b):
        return a * (1 - np.exp(-b*x))
    popt, _ = curve_fit(main_sequence, amplitude, velocity)
    x = np.linspace(min(amplitude), max(amplitude), 300)
    plt.plot(x, main_sequence(x, *popt), lw=3)

#%% Figure 1B

data = np.load('Fig1B.npy')
scatter_plot(data, [5, 20])

#%% Figure 1C

data = np.load('Fig1C.npy')
scatter_plot(data, [0.5, 2.5])

#%% Supplemental Figure 1

data = np.load('SuppFig1.npy')
saccade_sequence(data)

#%% Functions - MEG Data 

def ITC_plot(data, cluster, sensor, vmin, vmax):
    
    fmin=8
    fmax=13
    tmin=-0.2
    tmax=0
    GradPairRule = 1
    
    Sensor_List = data.ch_names
    
    if sensor == 'grad':
        grad_sens = mne.pick_types(data.info, meg="grad")
        cluster_grad_sens = grad_sens[cluster]
        cluster_grad_sens = [Sensor_List[i] for i in cluster_grad_sens]
        cluster_data = data.copy().pick_channels(cluster_grad_sens)
        # Combine the significance of each pair of gradiometers
        nGradPairs = int(len(cluster_grad)/2)
        cluster_grad_pairs = np.zeros(nGradPairs)
        for ipair in range(nGradPairs):
            grad1 = ipair*2
            grad2 = grad1+1
            # Both gradiometers are significant 
            if cluster_grad[grad1]==True and cluster[grad2]==True:
                cluster_grad_pairs[ipair]=1
            # Both gradiometers are non-significant
            elif cluster_grad[grad1]==False and cluster[grad2]==False:
                cluster_grad_pairs[ipair]=0
            # Only one gradiometer of the pair is significant
            elif cluster_grad[grad1] != cluster[grad2]:
                # Rule 1: Keep the pair
                if GradPairRule == 1: 
                    cluster_grad_pairs[ipair]=1
                # Rule 2: Reject the pair
                elif GradPairRule == 0:
                    cluster_grad_pairs[ipair]=0
        # Turn into Boolean 
        cluster_plot = np.array(cluster_grad_pairs, dtype='bool') 
        
    elif sensor == 'mag':
        mag_sens = mne.pick_types(data.info, meg="mag")
        cluster_mag_sens = mag_sens[cluster]
        cluster_mag_sens = [Sensor_List[i] for i in cluster_mag_sens]
        cluster_data = data.copy().pick_channels(cluster_mag_sens)
        cluster_plot = cluster
        
    # Time-Frquency Map
    cluster_data.plot(combine='mean', tmin=-0.4, tmax=0.4, fmin=4,vlim=[vmin, vmax], cmap='RdBu_r')
    # Topoplot with mask 
    data.plot_topomap(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, ch_type=sensor ,cmap='RdBu_r',vlim=(vmin, vmax),
                      mask=cluster_plot,mask_params=dict(marker='*', markerfacecolor='k', markeredgecolor='k',
                                                               linewidth=0, markersize=6))
    

#%% Figure 2 - Supplemental Figure 2

data = mne.time_frequency.read_tfrs('Fig2SuppFig2-tfr.h5')
# Figure 2 - Gradiometers 
data.plot(combine='mean', picks='grad', tmin=-0.4, tmax=0.4, fmin=4)
data.plot_topomap(fmin=8, fmax=13, tmin=-0.4, tmax=0.4, ch_type='grad', cmap='RdBu_r')
# Supplemental Figure 2 - Magnetometers
data.plot(combine='mean', picks='mag', tmin=-0.4, tmax=0.4, fmin=4)
data.plot_topomap(fmin=8, fmax=13, tmin=-0.4, tmax=0.4, ch_type='mag', cmap='RdBu_r')

#%% Supplemental Figure 3

data = mne.time_frequency.read_tfrs('Fig2SuppFig2-tfr.h5')
# Supplemental Figure 3: Figure 2 with baseline correction 
data.plot(combine='mean', picks='grad', tmin=-0.4, tmax=0.4, fmin=4, baseline=(-0.3,0))
# ERF
erf = mne.read_evokeds('SuppFig3.fif',verbose=False)
erf[0].plot(spatial_colors=True, picks=("grad"), xlim=[-0.4, 0.4])

#%% Figure 3 - Supplemental Figure 4

data =  mne.time_frequency.read_tfrs('Fig3SuppFig4-tfr.h5')
cluster_grad = np.load('Fig3.npy')
cluster_mag = np.load('SuppFig4.npy')

# Figure 3 - Gradiometers
ITC_plot(data, cluster_grad, 'grad', 0, 0.015)

# Supplemental Figure 4 - Magnetometers
ITC_plot(data, cluster_mag, 'mag', 0, 0.015)


#%% Figure 4 - Supplemental Figure 6

data =  mne.time_frequency.read_tfrs('Fig4SuppFig6-tfr.h5')
cluster_grad = np.load('Fig4.npy')
cluster_mag = np.load('SuppFig6.npy')

# Figure 4 - Gradiometers
ITC_plot(data, cluster_grad, 'grad', 0, 0.035)

# Supplemental Figure 6 - Magnetometers
ITC_plot(data, cluster_mag, 'mag', 0, 0.035)


#%% Supplemental Figure 5

# Supplemental Figure 5A,B - Correct
data =  mne.time_frequency.read_tfrs('SuppFig5AB-tfr.h5')
cluster_grad = np.load('SuppFig5A.npy')
ITC_plot(data, cluster_grad, 'grad', 0.06, 0.12)

# Supplemental Figure 5C,D - Inorrect
data =  mne.time_frequency.read_tfrs('SuppFig5CD-tfr.h5')
cluster_grad = np.load('SuppFig5C.npy')
ITC_plot(data, cluster_grad, 'grad', 0.06, 0.12)

#%% Figure 5 - Supplemental Figure 7

import nilearn.plotting
import nibabel as nib

t1_fname = 'T1.mgz'

# Figure 5A
data = nib.load('Fig5A.nii')
nilearn.plotting.plot_stat_map(data, str(t1_fname), draw_cross=False, black_bg=True, vmin=-0.014, vmax=0.025, 
           cut_coords=(35, 0, 45))

# Figure 5B
data = nib.load('Fig5B.nii')
nilearn.plotting.plot_stat_map(data, str(t1_fname), draw_cross=False, black_bg=True, vmin=-0, vmax=0.014, 
           cut_coords=(35, 0, 45))

# Supplemental Figure 7
data = nib.load('SuppFig7.nii')
nilearn.plotting.plot_stat_map(data, str(t1_fname), draw_cross=False, black_bg=True, vmin=-0.014, vmax=0.027, 
           cut_coords=(35, 0, 45))


