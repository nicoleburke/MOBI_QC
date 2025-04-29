import pyxdf
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import wave
#import pyaudio
import numpy as np
#import sounddevice as sd
from utils import *
import cv2
from scipy.signal import iirnotch, filtfilt
from glob import glob

events = {
    200: 'Onset_ExperimentStart',
    10: 'Onset_RestingState',
    11: 'Offset_RestingState',
    500: 'Onset_StoryListening',
    501: 'Offset_StoryListening',
    100: 'Onset_10second_rest',
    101: 'Offset_10second_rest', 
    20: 'Onset_CampFriend',
    21: 'Offset_CampFriend',
    30: 'Onset_FrogDissection',
    31: 'Offset_FrogDissection',
    40: 'Onset_DanceContest',
    41: 'Offset_DanceContest',
    50: 'Onset_ZoomClass',
    51: 'Offset_ZoomClass',
    60: 'Onset_Tornado',
    61: 'Offset_Tornado',
    70: 'Onset_BirthdayParty',
    71: 'Offset_BirthdayParty',
    300: 'Onset_subjectInput',
    301: 'Offset_subjectInput',
    302: 'Onset_FavoriteStory',
    303: 'Offset_FavoriteStory',
    304: 'Onset_WorstStory',
    305: 'Offset_WorstStory',
    400: 'Onset_impedanceCheck',
    401: 'Offset_impedanceCheck',
    80: 'Onset_SocialTask',
    81: 'Offset_SocialTask',
    201: 'Offset_ExperimentEnd',
}

story_onsets = [20, 30, 40, 50, 60, 70]

audiofiles= [
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Camp_Lose_A_Friend.wav",
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Frog_Dissection_Disaster.wav",
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/I_Decided_To_Be_Myself_And_Won_A_Dance_Contest.wav",
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/I_Fully_Embarrassed_Myself_In_Zoom_Class1.wav",
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Left_Home_Alone_in_a_Tornado.wav",
    "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/The_Birthday_Party_Prank_44.wav",
]

def load_behavior_data(xdf_filename):
    #xdf_filename = '/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
    stim_df = import_stim_data(xdf_filename)
    return stim_df

def get_secs_between_triggers(stim_df, trigger1, trigger2):
    return stim_df.loc[stim_df.trigger == trigger1, 'lsl_time_stamp'].values[0] - stim_df.loc[stim_df.trigger == trigger2, 'lsl_time_stamp'].values[0]

def missing_markers(events, stim_df):
    missing_markers=[]
    for event in events:
        if event in stim_df.event:
            return None
        else:
            missing_markers = missing_markers + [event]
        return missing_markers

def total_experiment_duration():
    minutes_entire_experiment, seconds = divmod(get_secs_between_triggers(201, 200), 60) 
    total_duration = f"{int(minutes_entire_experiment):02}:{int(seconds):02}"

    return total_duration

def story_listening_exp_durations():
    durations = pd.DataFrame({
    'trigger':story_onsets,
    'story':[events[x] for x in story_onsets],
    'lsl_duration': [get_secs_between_triggers(x+1, x) for x in story_onsets],
    'audiofile_duration': [wave.open(x).getnframes()/wave.open(x).getframerate() for x in audiofiles] #duration of audio file is number of frames divided by the frame rate.
    })

    durations['difference(sec)'] = durations['audiofile_duration'] - durations['lsl_duration']
    
    task_duration_diff = []

    # Calculating audiofile duration in 48kHz and then comparing with story listening durations from stim_df
    for i in range(len(durations.audiofile_duration)):
        task_duration = (durations.audiofile_duration[i] * 44100) / 48000
        if (durations.lsl_duration[i].round(3) -  task_duration.round(3)) > 0.5:
            task_duration_diff = task_duration_diff + [durations.story[i]]

    if task_duration_diff != []:
        list_task_duration_diff = task_duration_diff
    else:
        list_task_duration_diff = None

    return list_task_duration_diff

def rest_and_social_duration(trigger):
    trial_duration = get_secs_between_triggers(trigger+1, trigger)
    print(trial_duration)
    if trial_duration <= 305.0  and trial_duration >= 298.0:
        return True
    else:
        return False
    
def impedance_check_duration():
    impedance_check_mins, impedance_check_seconds = divmod(get_secs_between_triggers(401, 400), 60)
    impedance_duration = f"{int(impedance_check_mins):02}:{int(impedance_check_seconds):02}"

    return impedance_duration

def ten_secs_rest(stim_df):
    # Count the number of occurrences of trigger value 100 using sum
    evs = stim_df.loc[stim_df.event != 'psychopy_time_stamp']
    trigger_count = (evs['trigger'] == 100).sum()

    print(f"Number of trigg: {trigger_count}")

    #rest_onsets = 
    ten_secs_rest_durations = pd.DataFrame({
        'trigger':[x for x in range(trigger_count)],
        'story': ['Onset_10second_rest' for x in range(trigger_count)],
        'lsl_duration': [get_secs_between_triggers(x+1, x) for x in evs['trigger'] if x == 100]})
    #print(ten_secs_rest_durations)

    # Check if rest durations are equal
    equal_rest_durations = all(x == ten_secs_rest_durations['lsl_duration'][0] for x in ten_secs_rest_durations['lsl_duration'])
    
    return equal_rest_durations

def average_question_response_time(stim_df):
    response_times = []
    trial_response_times = []
    trigger_idx = 0
    for idx, x in enumerate(stim_df['trigger']):
        if idx in stim_df.loc[stim_df.trigger == 300,'lsl_time_stamp'].index:
            response_times = response_times + [stim_df.loc[stim_df.trigger == 301, 'lsl_time_stamp'].values[trigger_idx] - stim_df.loc[stim_df.trigger == 300, 'lsl_time_stamp'].values[trigger_idx]]
            trigger_idx = trigger_idx + 1
        elif x == 100 or x == 400:
            trial_response_times = trial_response_times + [sum(response_times)]
            response_times = []
    average_response_mins, average_response_seconds = divmod(sum(trial_response_times)/len(trial_response_times), 60)     
    average_response_time = f"{int(average_response_mins):02}:{int(average_response_seconds):02}"

    return average_response_time

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass