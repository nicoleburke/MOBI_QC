import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import librosa
import datetime
import matplotlib.pyplot as plt
from utils import *

def mic_lsl_wav_durations(xdf_filename:str, mic_df: pd.DataFrame) -> float:
    # get path of wav file in same folder 
    wav_path = glob(os.path.join(os.path.dirname(xdf_filename), '*.wav'))[0]
    
    # calculate wav and lsl durations
    wav_dur = round(librosa.get_duration(path=wav_path), 3)
    lsl_dur = round(mic_df['lsl_time_stamp'].iloc[-1]- mic_df['lsl_time_stamp'].iloc[0], 3)

    # diff
    dur_diff = abs(wav_dur - lsl_dur)

    return dur_diff

def mic_nans(mic_df: pd.DataFrame) -> tuple[int, float]:
    num_NaN = mic_df['int_array'].isna().sum()
    percent_NaN = num_NaN/len(mic_df)

    return num_NaN, percent_NaN

def mic_samples_stats(mic_df: pd.DataFrame) -> tuple[float, float, float, int, int]:
    quan25 = np.quantile(mic_df['int_array'], 0.25)
    quan75 = np.quantile(mic_df['int_array'], 0.75) 
    std = mic_df['int_array'].std()
    minn = min(mic_df.int_array)
    maxx = max(mic_df.int_array)

    return quan25, quan75, std, minn, maxx

def mic_plots(mic_df: pd.DataFrame, stim_df: pd.DataFrame):
    #hist
    plt.hist(mic_df['int_array'], bins=100)
    plt.xlabel('Microphone Samples')
    plt.ylabel('Count')
    plt.title('Distribution of Microphone Samples')
    plt.savefig(f'report_images/mic_histogram.png')

    #line plot
    plt.figure(figsize=(9, 3))
    plt.plot(mic_df.lsl_time_stamp, mic_df.int_array)
    for event in stim_df.loc[stim_df.event.str.contains('StoryListening|SocialTask')].iterrows():
        plt.axvline(event[1]['lsl_time_stamp'], color='r')
        plt.text(event[1]['lsl_time_stamp']+4, 0, event[1]['event'], rotation=90, verticalalignment='center', fontweight = 'bold')

    plt.xlabel('LSL Timestamps')
    plt.ylabel('Samples')
    plt.title('Microphone Samples over Time')
    plt.tight_layout()
    plt.savefig(f'report_images/mic_lineplot.png')


def mic_qc(xdf_filename:str) -> dict:
    # load data
    sub_id = xdf_filename.split('-')[1].split('/')[0]
    mic_df = import_mic_data(xdf_filename)
    stim_df = import_stim_data(xdf_filename)

    sampling_rate = get_sampling_rate(mic_df)

    vars = {}
    vars['sampling_rate'] = sampling_rate
    print(f"Effective sampling rate: {sampling_rate:.3f}")

    vars['lsl_wav_duration_diff'] = mic_lsl_wav_durations(xdf_filename, mic_df)
    print(f"Difference between .wav file and lsl timestamps durations: {vars['lsl_wav_duration_diff']:.3f}")

    vars['num_NaN'], vars['percent_NaN'] = mic_nans(mic_df)
    print(f"number of NaN's: {vars['num_NaN']} \npercent of NaN's: {vars['percent_NaN']:.3%}")
    vars['quan25'], vars['quan75'], vars['std'], vars['min'], vars['max'] = mic_samples_stats(mic_df)
    print('mic samples first quartile: {} \nmic samples third quartile: {}'.format(vars['quan25'], vars['quan75']))
    print('mic samples standard deviation: {:.3f}'.format(vars['std']))
    print(f"mic samples min: {vars['min']} \nmic samples max: {vars['max']}")
    
    mic_plots(mic_df, stim_df)

    return vars


# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass