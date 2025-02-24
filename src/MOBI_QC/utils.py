import pandas as pd
import pyxdf


def import_eyetracking_data(xdf_filename):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'ET'}])
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)

    return df



def get_stim(xdf_filename):
    stim, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name':'Stimuli_Markers'}])
    stim_df = pd.DataFrame(stim[0]['time_series'])
    stim_df.rename(columns={0: 'trigger'}, inplace=True)
    events = {
        200: 'Onset_Experiment',
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
        201: 'Offset_Experiment',
    }

    story_onsets = [20, 30, 40, 50, 60, 70]

    # relabel the event if the trigger is in the events dictionary, else if 
    stim_df['event'] = stim_df['trigger'].apply(lambda x: events[x] if x in events.keys() else 'Bx_input')

    # relabel the event as a psychopy timestamp if the trigger is greater than 5 digits
    stim_df.loc[stim_df.trigger.astype(str).str.len() > 5, 'event'] = 'psychopy_time_stamp'
    stim_df['lsl_time_stamp'] = stim[0]['time_stamps']
    stim_df['time'] = (stim[0]['time_stamps'] - stim[0]['time_stamps'][0])/1000
    stim_df
    return stim_df


def get_event_data(event, df, stim_df):
    """
    Get the data from the EEG dataframe that corresponds to the event in the stimuli dataframe.
    
    Args:
        event (str): The event to get the data for.
        df (pd.DataFrame): The EEG dataframe.
        stim_df (pd.DataFrame): The stimuli dataframe.
    
    Returns:
        pd.DataFrame: The  data corresponding to the event.
        """
    return df.loc[(df.lsl_time_stamp >= stim_df.loc[stim_df.event == 'Onset_'+event, 'lsl_time_stamp'].values[0]) & 
                  (df.lsl_time_stamp <= stim_df.loc[stim_df.event == 'Offset_'+event, 'lsl_time_stamp'].values[0])]

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass