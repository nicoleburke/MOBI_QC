import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import datetime
import re
import matplotlib.pyplot as plt
from utils import *

def et_val(et_df: pd.DataFrame) -> pd.DataFrame:
    # percent valid for all data columns (excluding time + validity columns)

    # remove columns w validity or time data 
    time_cols = et_df.filter(like = 'time').columns
    val_cols = et_df.filter(like = 'validity').columns
    qc_cols = time_cols.append(val_cols)
    et_data_cols = et_df.columns.drop(qc_cols)

    # percent non-NaN for each variable
    val_df = pd.DataFrame(columns= ['variable', 'percent_valid'])
    val_df['variable'] = et_data_cols

    for i, var in enumerate(et_data_cols):
        val_df.loc[i, 'percent_valid'] = round((1 - et_df[var].isna().mean()), 4)

    return val_df

def et_flag_1(val_df: pd.DataFrame) -> bool:
    # all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter)
    # compare coordinates (0,1,2) (validity within measures)
    val_flag1 = True
    for i in range(1, len(val_df)):
        # get variables that end in numbers 
        root = re.sub(r"_\d+$", "", val_df.loc[i, 'variable'])
        if root in val_df.loc[i-1, 'variable']:
            current_percent = val_df.loc[i, 'percent_valid']
            prev_percent = val_df.loc[i-1, 'percent_valid']
            if current_percent != prev_percent:
                print("ERROR: {} does not equal {}!".format(val_df.loc[i-1, 'variable'], val_df.loc[i, 'variable']))
                val_flag1 = False
    if val_flag1:
        print("all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter)")

    return val_flag1

def et_flag_2(val_df: pd.DataFrame) -> bool:
    # validity between coordinate systems 
    # all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter)

    val_flag2 = True
    for i in range(1, len(val_df)):
        root = val_df.loc[i, 'variable'].split("_in")[0]
        root_percent = val_df.loc[i, 'percent_valid']

        matching = val_df[val_df['variable'].str.contains(root)]

        for i in matching.index:
                matching_percent = matching.loc[i, 'percent_valid']
                matching_variable = matching.loc[i, 'variable']

                if root_percent != matching_percent:
                    print("ERROR: {} and {} were different by a difference of {}.".format(matching_variable, root, (root_percent-matching_percent)))
                    val_flag2 = False
    
    if val_flag2:
        print("% NaNs is the same between UCS and TBCS (gaze origin) and between UCS and display area (gaze point)")

    val_flag2

def et_val_LR(val_df: pd.DataFrame) -> float:
    # compare valid data between left and right eyes
    left = val_df[val_df.variable.str.startswith('left')]
    right = val_df[val_df.variable.str.startswith('right')]

    RL_val = pd.DataFrame(columns = ['eye','min', 'max', 'mean'])

    for i, (df, RL) in enumerate([(left, 'left'), (right, 'right')]):
        min1 = min(df['percent_valid'])
        max1 = max(df['percent_valid'])
        mean1 = round(np.mean(df['percent_valid']), 4)
        RL_val.loc[i] = [RL, min1, max1, mean1]

    # find diff between RL 
    RL_val.loc[2] = ['diff', RL_val['min'].diff()[1], RL_val['max'].diff()[1], RL_val['mean'].diff()[1]]

    # add blank row 
    blank = pd.DataFrame([['', '', '', '']], columns = RL_val.columns)
    RL_val = pd.concat([RL_val.iloc[:2], blank, RL_val.iloc[2:]])
    RL_val.reset_index(drop=True, inplace=True) 

    # mean 
    lmean = RL_val.loc[RL_val.eye =='left', 'mean'][0]
    rmean = RL_val.loc[RL_val.eye =='right', 'mean'][1]
    mean_diff = RL_val.loc[RL_val.eye =='diff', 'mean'][3]

    return abs(mean_diff)

def et_percent_over02(et_df: pd.DataFrame) -> float:
    # distance between gaze points
    # remove NaNs
    et_nums = et_df[~np.isnan(et_df.left_gaze_point_on_display_area_0) &
            ~np.isnan(et_df.left_gaze_point_on_display_area_1) &
            ~np.isnan(et_df.right_gaze_point_on_display_area_0) &
            ~np.isnan(et_df.right_gaze_point_on_display_area_1)]
    
    # distribution of distance between gaze points
    x1 = et_nums.right_gaze_point_on_display_area_0
    x2 = et_nums.left_gaze_point_on_display_area_0
    y1 = et_nums.right_gaze_point_on_display_area_1
    y2 = et_nums.left_gaze_point_on_display_area_1
    dists = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    percent_over02 = sum(dists >= 0.2)/len(dists)  
    return percent_over02 

def et_lineplot(et_df: pd.DataFrame, percent_over02: float, sub_id: str):
    # calculate distances including NaNs
    x1 = et_df.right_gaze_point_on_display_area_0
    x2 = et_df.left_gaze_point_on_display_area_0
    y1 = et_df.right_gaze_point_on_display_area_1
    y2 = et_df.left_gaze_point_on_display_area_1
    all_dists = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # plt with x = index 
    plt.figure(figsize=(10, 3))
    plt.plot(all_dists)
    plt.title(f"Distance Between Left and Right Gaze Points Over Time")
    plt.axhline(y = 0.2, color = 'red', label = 'Gaze point difference = 0.2')
    plt.ylabel("Gaze Point Difference (mm)")
    plt.xlabel("Time (s)")
    plt.legend()
    print(f"percent over 0.20: {percent_over02}%")
    plt.tight_layout()
    plt.savefig(f'report_images/{sub_id}_et_gazedifference.png')


def et_qc(xdf_filename: str):
    sub_id = xdf_filename.split('-')[1].split('/')[0]
    et_df = import_et_data(xdf_filename)

    sampling_rate = get_sampling_rate(et_df)
    val_df = et_val(et_df)

    vars = {}

    vars['sampling_rate'] = sampling_rate
    print(f"Effective sampling rate: {sampling_rate:.3f}")

    vars['flag1'] = et_flag_1(val_df)
    print(f'Flag: all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter): {vars['flag1']}')

    vars['flag2'] = et_flag_2(val_df)
    print(f'Flag: % of NaNs is the same between coordinate systems (UCS and TBCS (gaze origin) and between UCS and display area (gaze point)): {vars['flag2']}')

    vars['LR_mean_diff'] = et_val_LR(val_df)
    print(f'Mean difference in percent valid data between right and left eyes: {vars['LR_mean_diff']:.3%}')

    vars['percent_over02'] = et_percent_over02(et_df)
    print(f'Percent of data with gaze point differences of over 0.2 mm: {vars['percent_over02']:.3%}')

    et_lineplot(et_df, vars['percent_over02'], sub_id)

    return vars

   



    return vars