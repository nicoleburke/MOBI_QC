#%%

import eeg_research.preprocessing.pipelines.rockland_sample_cleaning_pipeline as pp
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyxdf
import pyprep

# %%
xdf_filename = '/Users/bryan.gonzalez/CUNY_subs/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
eeg, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type':'EEG'}])
ch_names = [f"E{i+1}" for i in range(128)]
units = {channel: 'volts' for channel in ch_names}
sfreq = 1000
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg[0]['time_series'].T * 1e-6, info=info)

# %%
# make a 1d array of zeros
value = np.zeros((1, raw.n_times))

info = mne.create_info(["Cz"], raw.info['sfreq'], ch_types='eeg')
cz = mne.io.RawArray(value, info)
raw.add_channels([cz], force_update_info=True)

# %%
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
raw.set_montage(montage, on_missing='ignore')
raw.crop(tmin=0, tmax=300)
#%%
prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
prep = pyprep.PrepPipeline(raw, montage=montage, channel_wise=True, prep_params=prep_params)

# %%
prep_output = prep.fit()
# %%
def annotate_blinks(
    raw: mne.io.Raw, ch_name: list[str] = ["Fp1", "Fp2"]
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
