#%%
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyprep
import pyxdf
from utils import *
from scipy.signal import welch
import warnings
warnings.filterwarnings("ignore")

#%%

xdf_filename = '/Users/bryan.gonzalez/CUNY_subs/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
def compute_eeg_pipeline(xdf_filename):
    """
    This function computes the EEG pipeline for the given xdf file.
    Args:
        xdf_filename (str): The path to the xdf file.
    """    
    subject = xdf_filename.split('-')[1].split('/')[0]
    df = get_event_data(event='RestingState', 
                        df=import_eeg_data(xdf_filename),
                        stim_df=import_stim_data(xdf_filename))

    
    ch_names = [f"E{i+1}" for i in range(df.shape[1] - 1)]
    info = mne.create_info(ch_names, 
                        sfreq=1/df.lsl_time_stamp.diff().mean(), 
                        ch_types='eeg')
    df.drop(columns=['lsl_time_stamp'], inplace=True)

    raw = mne.io.RawArray(df.T * 1e-6, info=info) # multiplying by 1e-6 converts to volts

    # Create a Cz reference
    value = np.zeros((1, raw.n_times))
    info = mne.create_info(["Cz"], raw.info['sfreq'], ch_types='eeg')
    cz = mne.io.RawArray(value, info)
    raw.add_channels([cz], force_update_info=True)

    # Apply a montage
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    raw.set_montage(montage, on_missing='ignore')

    
    prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
        }
    # these params set up the robust reference  - i.e. median of all channels and interpolate bad channels
    prep = pyprep.PrepPipeline(raw, montage=montage, channel_wise=True, prep_params=prep_params)
    prep_output = prep.fit()
    raw_cleaned = prep_output.raw_eeg

    
    vars = {}
    print(f"Bad channels before robust reference: {prep.noisy_channels_original['bad_all']}")
    vars['bad_channels_before'] = prep.noisy_channels_original['bad_all']
    print(f"Interpolated channels: {prep.interpolated_channels}")
    vars['interpolated_channels'] = prep.interpolated_channels
    print(f"Bad channels after interpolation: {prep.still_noisy_channels}")
    vars['bad_channels_after'] = prep.still_noisy_channels
    
    fig = raw_cleaned.plot_psd(tmax=np.inf, fmax=250)
    fig.savefig(f'report_images/{subject}_eeg_psd.png')
    
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
    
    fig = raw_cleaned.plot(show_scrollbars=False,
                        show_scalebars=False,events=None, start=0, duration=300,n_channels=75, scalings=11e-5,color='k')
    fig.grab().save(f'report_images/{subject}_eeg_annotations.png')
    
    # Applying a high pass filter to remove low frequency noise
    raw_cleaned.filter(l_freq=0.5, h_freq=None)

    blink_annotations = annotate_blinks(raw_cleaned, ch_name=["E25", "E8"])

    muscle_annotations = annotate_muscle(raw_cleaned)

    all_annotations = blink_annotations + muscle_annotations + raw.annotations
    raw_cleaned.set_annotations(all_annotations)
    
    # Create a binary array
    binary_mask = np.zeros(len(raw_cleaned.times), dtype=int)

    # Iterate over annotations
    for annot in raw_cleaned.annotations:
        onset_sample = int(annot['onset'] * raw_cleaned.info['sfreq'])
        duration_sample = int(annot['duration'] * raw_cleaned.info['sfreq'])
        binary_mask[onset_sample:onset_sample + duration_sample] = 1

    percent_good = 1 - np.sum(binary_mask) / len(binary_mask)
    print(f'Percent Good Data: {percent_good * 100:.2f}%')
    vars['percent_good'] = percent_good * 100
    
    #filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    filt_raw = raw_cleaned.copy()
    filt_raw.info['bads'] = prep.still_noisy_channels 
    ica = mne.preprocessing.ICA(n_components=.99, method='picard')
    ica.fit(filt_raw)
    #ica.plot_sources(filt_raw)
    ica.plot_components()#.savefig(f'report_images/{subject}_ica_components.png')
    plt.savefig(f'report_images/{subject}_ica_components.png')
    comp_idx, scores = ica.find_bads_muscle(filt_raw)

    # Remove the muscle artifacts
    raw_ica = ica.apply(filt_raw, exclude=comp_idx)

    
    eog_evoked = mne.preprocessing.create_eog_epochs(raw_cleaned, ch_name=['E8', 'E25']).average(picks="all")
    eog_evoked.apply_baseline((None, None))
    eog_evoked.plot_joint().savefig(f'report_images/{subject}_eog_evoked.png')
    
    eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_cleaned, n_eeg=1, reject=None, no_proj=True,
                                                    ch_name=['E8', 'E25'])

    return vars, raw_cleaned


def test_eeg_pipeline(xdf_filename):
    """
    Test the EEG pipeline for the given xdf file.
    Args:
        xdf_filename (str): The path to the xdf file.
    """
    vars, raw_cleaned = [43, [1,2,3]]#compute_eeg_pipeline(xdf_filename)
    print(vars)
    #print(raw_cleaned.info['bads'])
    return vars, raw_cleaned

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass
# %%

#vars = compute_eeg_pipeline(xdf_filename)

# %%
