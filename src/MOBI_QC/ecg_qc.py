import pyxdf
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import wave
#import pyaudio
import numpy as np
#import sounddevice as sd
from utils import *
import scipy
from scipy.signal import iirnotch, filtfilt
from glob import glob
import neurokit2 as nk
from neurokit2.signal import signal_power

def get_modality(xdf_filename):
    #xdf_filename = '/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
    subject = xdf_filename.split('-')[1].split('/')[0]
    ps_df = get_event_data(event='RestingState',
                    df=import_physio_data(xdf_filename),
                    stim_df=import_stim_data(xdf_filename))
    ecg_df = ps_df[['ECG1', 'lsl_time_stamp', 'time']]
    return ecg_df, subject

def ecg_sampling_rate(ecg_df):
    effective_sampling_rate = 1 / (ecg_df.lsl_time_stamp.diff().mean())
    return effective_sampling_rate

def ecg_preprocess(ecg_df):
    ecg_signals, info = nk.ecg_process(ecg_df['ECG1'], sampling_rate=ecg_sampling_rate(ecg_df), method='neurokit')
    return ecg_signals, info

def average_heartrate(ecg_signals):
    avg_heartrate = sum(ecg_signals['ECG_Rate'])/len(ecg_signals['ECG_Rate'])
    return avg_heartrate

def ecg_quality_kurtosis_SQI(ecg_cleaned, method="fisher"):
    """Return the kurtosis of the signal, with Fisher's or Pearson's method."""

    if method == "fisher":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=True)
    elif method == "pearson":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=False)

def ecg_quality_psd_SQI(
    ecg_cleaned,
    sampling_rate,
    window=1024,
    num_spectrum=[5, 15],
    dem_spectrum=[5, 40],
    **kwargs
):
    """Power Spectrum Distribution of QRS Wave."""

    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]

    return num_power / dem_power

def ecg_quality_baseline_power_SQI(
    ecg_cleaned,
    sampling_rate,
    window=1024,
    num_spectrum=[0, 1],
    dem_spectrum=[0, 40],
    **kwargs
):
    """Relative Power in the Baseline."""
    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]

    return (1 - num_power) / dem_power

def ecg_snr(ecg_signals, ecg_df):
    duration = len(ecg_df['ECG1'].tolist()) / ecg_sampling_rate(ecg_df)
    t = np.linspace(0, duration , len(ecg_df['ECG1']))
    t = t[:5000]

    # Clean ECG from ecg_signals dataframe
    ecg_cleaned = ecg_signals['ECG_Clean'] 

    # Calculate signal power (variance of the cleaned ECG signal)
    signal_power = np.var(ecg_cleaned)

    # Estimate noise power (using residual noise after subtracting cleaned signal from raw noisy signal)
    noise_signal = ecg_signals['ECG_Raw'] - ecg_cleaned  # residual noise
    noise_power = np.var(noise_signal)

    # Calculate SNR (Signal-to-Noise Ratio in dB)
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def ecg_report_plot(ecg_signals, info, subject):
    nk.ecg_plot(ecg_signals, info)
    fig = plt.gcf()
    axes = fig.get_axes()
    fig.set_size_inches(25, 10)
    plt.tight_layout()

    # Iterate over each axis and move the legend
    for i, ax in enumerate(axes):
        if i == 0: 
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend outside the plot
        elif i == 1:  
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend outside the plot
        elif i == 2:  
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend to the right
        else:
            ax.legend(loc='center right', bbox_to_anchor=(1, 0.5), fontsize=15)  # Default position for other plots
    for ax in axes:
        ax.grid(True)

    # Save plot
    plt.savefig(f'report_images/{subject}_ecg_report.png')
    plt.show()

    return plt

if __name__ == "__main__":
    pass
