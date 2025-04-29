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
from scipy.signal import butter, filtfilt


xdf_filename = '/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
subject = xdf_filename.split('-')[1].split('/')[0]
ps_df = get_event_data(event='Experiment',
                    df=import_physio_data(xdf_filename),
                    stim_df=import_stim_data(xdf_filename))

eda_df = ps_df[['EDA2', 'lsl_time_stamp', 'time']]

def eda_sampling_rate(eda_df):
    effective_sampling_rate = 1 / (eda_df.lsl_time_stamp.diff().mean())
    return effective_sampling_rate

# Preprocess EDA signal
eda_signals, info = nk.eda_process(eda_df['EDA2'], sampling_rate=eda_sampling_rate(eda_df), method='neurokit')

# Checking for nan or missing values in EDA data and return a percentage validity
def eda_signal_integrity_check(eda_df):
    count_nan = 0
    for x in eda_df['EDA2']:
        if np.isnan(x) == True:
            count_nan = count_nan + 1

    eda_validity = 100 - (count_nan/len(eda_df['EDA2'])) * 100
    return eda_validity

def eda_preprocess(eda_df):
    # Preprocess EDA signal
    eda_signals, info = nk.eda_process(eda_df['EDA2'], sampling_rate=eda_sampling_rate(eda_df), method='neurokit')
    return eda_signals, info

def scl_stability(scl):
    # Calculating average, standard deviation and coefficient of variation

    average_scl = np.mean(scl)
    scl_sd = np.std(scl)
    scl_cv = (scl_sd / average_scl) * 100
    return average_scl, scl_sd, scl_cv

def scl_trend_analysis(eda_signals):

    # Calculating rolling mean of SCL and slope of rolling mean of SCL over time

    # Calculating slope of SCL over time

    scl_df = pd.DataFrame(eda_signals['EDA_Tonic'])
    scl_df['lsl_time_stamp'] = eda_df['lsl_time_stamp']

    # Calculating slope of SCL 
    scl_df['EDA_Tonic_Slope'] = np.gradient(scl_df['EDA_Tonic'], eda_df['lsl_time_stamp'])
    
    # Calculating rolling mean of SCL and Slope of SCL rolling mean
    rolling_mean = pd.Series(eda_signals['EDA_Tonic']).rolling(window=(int)(eda_sampling_rate(eda_df)), center=True).mean()
    slope_rolling_mean = np.gradient(rolling_mean)
    
    plt.figure(figsize=(20,5))
    plt.plot(eda_df['lsl_time_stamp'], slope_rolling_mean, label='SCL Rolling_mean slope', color='orange', linestyle='-')
    plt.plot(eda_df['lsl_time_stamp'], scl_df['EDA_Tonic_Slope'], label='Slope of SCL', color='blue')
    plt.title('SCL Slope and Rolling Mean Slope Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Slope', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(f'report_images/{subject}_eda_slope.png',dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

    return plt

def scr_amplitudes(info):
    # Calculating average amplitude of SCRs
    scr_amplitudes = [amplitude for amplitude in info['SCR_Amplitude'] if np.isnan(amplitude) != True]
    average_scr_amplitude = np.mean(scr_amplitudes)

    # Calculating SCR amplitude Validity
    count_invalid_scr = 0
    for amplitude in scr_amplitudes:
        if amplitude > 0.01 and amplitude < 3.0:
            continue
        else:
            count_invalid_scr = count_invalid_scr + 1

    scr_amplitude_validity = 100 - (count_invalid_scr/len(scr_amplitudes)) * 100

    return average_scr_amplitude, scr_amplitude_validity

def eda_snr(eda_df):
    duration = len(eda_df['EDA2'].tolist()) / eda_sampling_rate(eda_df)
    t = np.linspace(0, duration , len(eda_df['EDA2']))
    t = t[:5000]

    # Clean ECG from ecg_signals dataframe
    eda_cleaned = eda_signals['EDA_Clean']

    # Calculate signal power (variance of the cleaned ECG signal)
    signal_power = np.var(eda_cleaned)

    # Estimate noise power (using residual noise after subtracting cleaned signal from raw noisy signal)
    noise_signal = eda_signals['EDA_Raw'] - eda_cleaned  # residual noise
    noise_power = np.var(noise_signal)

    # Calculate SNR (Signal-to-Noise Ratio in dB)
    snr = 10 * np.log10(signal_power / noise_power)

    # Output the results
    print(f"Signal Power: {signal_power}")
    print(f"Noise Power: {noise_power}")
    print(f"SNR: {snr} dB")

    return snr


def eda_report_plot(eda_signals, info):
    fig = nk.eda_plot(eda_signals, info)
    fig = plt.gcf()
    axes = fig.get_axes()
    fig.set_size_inches(20, 10)
    raw_signal_line = axes[0].lines[0]
    raw_signal_line.set_color('red')

    handles, labels = axes[0].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == "EDA_Raw":  
            handle.set_color('red')  

    axes[0].legend(handles, labels)  
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    plt.savefig(f'report_images/{subject}_eda_report.png')
    plt.show()

    return plt

if __name__ == "__main__":
    pass
