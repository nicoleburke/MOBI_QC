#%%

"""
This module provides functions and a pipeline for preprocessing and analyzing EEG data.

It includes functions for loading EEG data from XDF files, creating MNE Raw objects,
annotating blinks and muscle artifacts, and performing ICA to identify and remove artifacts.
    annotate_blinks(raw: mne.io.Raw, ch_name: list[str] = ["Fp1", "Fp2"]) -> mne.Annotations:
        Annotate the blinks in the EEG signal using specified channels.
    annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
        Annotate muscle artifacts in the EEG signal using z-score thresholding.
Usage:
    Load EEG data from an XDF file, preprocess it using the PrepPipeline, 
    annotate blinks and muscle artifacts, and perform ICA to identify and 
    remove artifacts.
    Load EEG data from an XDF file, preprocess it using the PrepPipeline, annotate blinks and muscle artifacts,
    and perform ICA to identify and remove artifacts.

Example:
    xdf_filename = '/path/to/your/file.xdf'
"""

import eeg_research.preprocessing.pipelines.rockland_sample_cleaning_pipeline as pp
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyprep
import pyxdf
from utils import *
from scipy.signal import welch

# %%
xdf_filename = '/Users/bryan.gonzalez/CUNY_subs/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
#%%

df = get_event_data(event='RestingState', 
                    df=import_eeg_data(xdf_filename),
                    stim_df=get_stim(xdf_filename))
#%%
# Make the time stamps column the index
df.set_index('lsl_time_stamp', inplace=True)

#%%
metrics = {}
sampling_rate = 1/df.index.to_series().diff().mean()

for channel in df.columns:

    if channel == 'lsl_time_stamp':
        continue

    signal =  df[channel].values
    signal = signal * 1e6 # convert to microvolts

    # Peak-to-peak amplitude
    peak_to_peak = np.ptp(signal)

    # Compute the absolute voltage range
    abs_range = np.max(signal) - np.min(signal)
    # Compute standard deviation of amplitude
    std = np.std(signal)


    # Check for flat channels that have zero variance
    is_flat = np.var(signal) < 1e6 # threshold for flatness

    # compute power spectral density using Welch's method
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)))

    # powerline noise
    powerline_freq = 60 if sampling_rate >= 120 else 50
    powerline_power = np.sum(psd[(freqs >= powerline_freq - 1) & (freqs <= powerline_freq + 1)])

    # High frequency noise
    hf_noise = np.sum(psd[freqs >= 100] if np.any(freqs >= 100) else np.nan)
    # Compute low frequency noise
    lf_noise = np.sum(psd[freqs <= .5] if np.any(freqs <= .5) else np.nan)
    # Compute alpha power
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)] if np.any((freqs >= 8) & (freqs <= 13)) else np.nan)
    


    # Signal-to-noise ratio
    signal_power = np.sum(psd[(freqs >= .5) & (freqs <= 40)]) # the EEG range
    noise_power = np.sum(psd[freqs > 40]) + 1e-6 # avoid division by zero
    snr = 10 * np.log10(signal_power / noise_power)

    # compute spectral entropy
    valid_idx = (freqs >= .5) & (freqs <= 40)
    psd = psd[valid_idx]
    freqs = freqs[valid_idx]
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm) + 1e10) #avoid log(0)

    metrics[channel] = {"peak_to_peak": peak_to_peak,
                        "is_flat": is_flat,
                        "powerline_power": powerline_power,
                        "hf_noise": hf_noise,
                        "lf_noise": lf_noise,
                        "alpha_power": alpha_power,
                        "spectral_entropy": spectral_entropy,
                        "snr": snr}

quality_df = pd.DataFrame.from_dict(metrics, orient='index')

# Compute dropped samples
timestamps = df.index
time_diffs = np.diff(timestamps)
expected_diff = 1/sampling_rate
dropped_samples = np.sum(time_diffs > expected_diff * 1.5)

# compute % of dropouts
prcnt_dropout = dropped_samples / len(timestamps) * 100


quality_df['dropped_samples'] = dropped_samples
quality_df['prcnt_dropout'] = prcnt_dropout



#%%

df = get_event_data(event='RestingState', 
                    df=import_eeg_data(xdf_filename),
                    stim_df=get_stim(xdf_filename))

ch_names = [f"E{i+1}" for i in range(df.shape[1] - 1)]
info = mne.create_info(ch_names, 
                       sfreq=1/df.lsl_time_stamp.diff().mean(), 
                       ch_types='eeg')
df.drop(columns=['lsl_time_stamp'], inplace=True)

raw = mne.io.RawArray(df.T * 1e-6, info=info) # multiplying by 1e-6 converts to volts

# %%
# make a 1d array of zeros
value = np.zeros((1, raw.n_times))

info = mne.create_info(["Cz"], raw.info['sfreq'], ch_types='eeg')
cz = mne.io.RawArray(value, info)
raw.add_channels([cz], force_update_info=True)

# %%
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
raw.set_montage(montage, on_missing='ignore')
#raw.crop(tmin=0, tmax=300)
#%%
prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
# these params set up the robust reference  - i.e. median of all channels and interpolate bad channels
prep = pyprep.PrepPipeline(raw, montage=montage, channel_wise=True, prep_params=prep_params)

# %%
prep_output = prep.fit()
# %%
def annotate_blinks(
    raw: mne.io.Raw, ch_name: list[str] = ["E25", "E8"]
) -> mne.Annotations:
    """Annotate the blinks in the EEG signal.
 
    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.
        ch_name (list[str]): The channels to use for the EOG. Default is
                             ["Fp1", "Fp2"]. I would suggest to use the
                             channels that are the most frontal (just above
                             the eyes). In the case of an EGI system the
                             channels would be "E25" and "E8".
 
    Returns:
        mne.Annotations: The annotations object containing the blink events.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=ch_name)
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
    )
    return blink_annotations

def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
        raw, 
        threshold=3, # this needs to be calibrated for the entire dataset
        ch_type='eeg', 
        min_length_good=0.1, 
        filter_freq=(95, 120), 
        )
 
    return muscle_annotations
#%%
print(f"Bad channels: {prep.interpolated_channels}")
print(f"Bad channels original: {prep.noisy_channels_original['bad_all']}")
print(f"Bad channels after interpolation: {prep.still_noisy_channels}")
#%%
raw_cleaned = prep_output.raw_eeg
#%%
# apply a lowpass filter to remove high frequency noise
raw_cleaned.filter(0.5, None)

blink_annotations = annotate_blinks(raw_cleaned, ch_name=["E25", "E8"])

muscle_annotations = annotate_muscle(raw_cleaned)

# %%
all_annotations = blink_annotations + muscle_annotations + raw.annotations
raw_cleaned.set_annotations(all_annotations)

# Then you can visually check to see if the annotations are correct. If it misses some blinks, apply a lowpass


# %%
# Create a binary array
binary_mask = np.zeros(len(raw_cleaned.times), dtype=int)

# Iterate over annotations
for annot in raw_cleaned.annotations:
    onset_sample = int(annot['onset'] * raw_cleaned.info['sfreq'])
    duration_sample = int(annot['duration'] * raw_cleaned.info['sfreq'])
    binary_mask[onset_sample:onset_sample + duration_sample] = 1

percent_good = 1 - np.sum(binary_mask) / len(binary_mask)
print(percent_good)


#%%
# PSD
psds, freqs = mne.time_frequency.psd_array_multitaper(raw_cleaned, sfreq=1000, fmin=.5, fmax=80, n_jobs=1)
psds = 10 * np.log10(psds)
psds_mean = psds.mean(0)
psds_std = psds.std(0)
f, ax = plt.subplots()
ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=.5)
ax.set(title='Multitaper PSD', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()

#%%
plt.figure(figsize=(15, 5))
raw_cleaned.plot(duration=5, scalings='auto', show=False, block=True)
plt.savefig("raw_plot.png", dpi=300)
plt.close()
#%%
# Quantifying blinks and muscles through ICA

ica = mne.preprocessing.ICA(n_components=None, method='picard')
ica.fit(raw_cleaned)
ica.plot_sources(raw_cleaned)
#%%
comp_idx, scores = ica.find_bads_muscle(raw_cleaned)
# scores represent the correlation between the ICs and the muscle artifacts
# % of components that are muscle artifacts
#len(comp_idx) / len(scores) * 100

# %%
# this would remove the muscle artifacts
raw_ica = ica.apply(raw_cleaned, exclude=comp_idx)
raw_ica.plot()
#%%
# %%
eog_evoked = mne.preprocessing.create_eog_epochs(raw_cleaned, ch_name=['E8', 'E25']).average(picks="all")
eog_evoked.apply_baseline((None, None))
eog_evoked.plot_joint()
# %%
eog_projs, _ = mne.preprocessing.compute_proj_eog(
    raw_cleaned, n_grad=1, n_mag=1, n_eeg=1, reject=None, no_proj=True
)

# %%
mne.viz.plot_projs_topomap(eog_projs, info=raw_cleaned.info)
# %%

# log power spectral density
raw_cleaned.plot_psd()

# %%
# make it log power spectral density
# Apply a notch filter to remove 60Hz noise
raw_cleaned.notch_filter(np.arange(60, 180, 60), filter_length='auto', phase='zero')
raw_cleaned.plot_psd(fmax=80, average=False)
# %%
# show 6Hz frequency in terms of scalp topography
ica.plot_sources(raw_cleaned, show_scrollbars=False)
# plot the components
ica.plot_components()

# %%
# plot the scalp topography of the different frequency bands
ica.plot_properties(raw_cleaned)

# %%


def eeg_qc():
    xdf_filename = '/Users/bryan.gonzalez/CUNY_subs/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'


    df = get_event_data(event='RestingState', 
                        df=import_eeg_data(xdf_filename),
                        stim_df=get_stim(xdf_filename))
    #%%
    # Make the time stamps column the index
    df.set_index('lsl_time_stamp', inplace=True)

    metrics = {}
    sampling_rate = 1/df.index.to_series().diff().mean()

    for channel in df.columns:

        if channel == 'lsl_time_stamp':
            continue

        signal =  df[channel].values
        signal = signal * 1e6 # convert to microvolts

        # Peak-to-peak amplitude
        peak_to_peak = np.ptp(signal)

        # Compute the absolute voltage range
        abs_range = np.max(signal) - np.min(signal)
        # Compute standard deviation of amplitude
        std = np.std(signal)


        # Check for flat channels that have zero variance
        is_flat = np.var(signal) < 1e6 # threshold for flatness

        # compute power spectral density using Welch's method
        freqs, psd = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)))

        # powerline noise
        powerline_freq = 60 if sampling_rate >= 120 else 50
        powerline_power = np.sum(psd[(freqs >= powerline_freq - 1) & (freqs <= powerline_freq + 1)])

        # High frequency noise
        hf_noise = np.sum(psd[freqs >= 100] if np.any(freqs >= 100) else np.nan)
        # Compute low frequency noise
        lf_noise = np.sum(psd[freqs <= .5] if np.any(freqs <= .5) else np.nan)
        # Compute alpha power
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)] if np.any((freqs >= 8) & (freqs <= 13)) else np.nan)
        


        # Signal-to-noise ratio
        signal_power = np.sum(psd[(freqs >= .5) & (freqs <= 40)]) # the EEG range
        noise_power = np.sum(psd[freqs > 40]) + 1e-6 # avoid division by zero
        snr = 10 * np.log10(signal_power / noise_power)

        # compute spectral entropy
        valid_idx = (freqs >= .5) & (freqs <= 40)
        psd = psd[valid_idx]
        freqs = freqs[valid_idx]
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm) + 1e10) #avoid log(0)

        metrics[channel] = {"peak_to_peak": peak_to_peak,
                            "is_flat": is_flat,
                            "powerline_power": powerline_power,
                            "hf_noise": hf_noise,
                            "lf_noise": lf_noise,
                            "alpha_power": alpha_power,
                            "spectral_entropy": spectral_entropy,
                            "snr": snr}

    quality_df = pd.DataFrame.from_dict(metrics, orient='index')

    # Compute dropped samples
    timestamps = df.index
    time_diffs = np.diff(timestamps)
    expected_diff = 1/sampling_rate
    dropped_samples = np.sum(time_diffs > expected_diff * 1.5)

    # compute % of dropouts
    prcnt_dropout = dropped_samples / len(timestamps) * 100


    quality_df['dropped_samples'] = dropped_samples
    quality_df['prcnt_dropout'] = prcnt_dropout

