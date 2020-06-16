#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:53:21 2020

@author: wasilaq
"""

'''
         pred
        |  0  |  1 |
     ___________________
true  0 |  TN | FP |
      1 |  FN | TP |
'''

import pandas as pd
import seaborn as sns


emotion_subset = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df_subset')

# group emotions
updated_subset = emotion_subset.copy()
updated_subset.loc[updated_subset['emotion'] == 'joyous','emotion'] = 'happy'
updated_subset.loc[updated_subset['emotion'] == 'relax','emotion'] = 'calm'
updated_subset.loc[updated_subset['emotion'] == 'soothing','emotion'] = 'calm'
updated_subset.loc[updated_subset['emotion'] == 'pain','emotion'] = 'sad'


# dataframe containing y values
identifiers = ['track_id','song_id','artist','song_title']

targets = pd.DataFrame()
targets['happy'] = updated_subset['emotion']=='happy'
targets['energetic'] = updated_subset['emotion']=='energetic'
targets['calm'] = updated_subset['emotion']=='calm'
targets['sad'] = updated_subset['emotion']=='sad'

targets.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/targets')

# EDA
def add_columns(first_df, second_df, columns):
    for column in columns:
        first_df[column] = second_df[column].astype(int)
    return first_df

X1 = emotion_subset.drop(identifiers+['emotion'], axis=1)
emotions = ['happy','energetic','calm','sad']
look1 = X1.copy()
look1 = add_columns(look1, targets, emotions)
look1.corr()
sns.pairplot(look1.drop(columns=['sad','energetic','calm'])) # happy
sns.pairplot(look1.drop(columns=['happy','energetic','calm'])) # sad
sns.pairplot(look1.drop(columns=['sad','happy','calm'])) # energetic
sns.pairplot(look1.drop(columns=['sad','energetic','happy'])) # calm