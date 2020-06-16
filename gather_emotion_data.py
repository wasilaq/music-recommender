#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:04:36 2020

@author: wasilaq
"""
from time import sleep
from requests.exceptions import ReadTimeout
from spotify_api_functions import audio_analysis
import numpy as np
import pandas as pd
import sqlite3


# identify song tags that correspond with each emotion
emotions = ['happy', 'joyous', 'energetic', 'soothing', 'relax', 'calm', 'sad', 'pain']
emotion_tags = {}

for emotion in emotions:
    loc = 'https://gaurav22verma.github.io/assets/IMACDataset/Tags/{}.txt'.format(emotion)
    tags = pd.read_csv(loc, names=['tag'])
    emotion_tags[emotion] = tags
    
emotion_tags.keys()
emotion_tags['pain'].head()


# create dataframe from db
def sanitize(tag):
    """
    sanitize a tag so it can be included or queried in the db
    """
    tag = tag.replace("'","''")
    return tag

conn = sqlite3.connect('/Users/wasilaq/Downloads/lastfm_tags.db')

emotions_df = pd.DataFrame()

for emotion in emotions:
    emotion_tid = set()
    for tag in list(emotion_tags[emotion]['tag']):
        sql = "SELECT tids.tid FROM tid_tag, tids, tags WHERE tids.ROWID=tid_tag.tid AND tid_tag.tag=tags.ROWID AND tags.tag='%s'" % sanitize(tag)
        res = conn.execute(sql)
        data = res.fetchall()
        emotion_tid.update([item[0] for item in data])
    df = pd.DataFrame(emotion_tid, columns=['track_id'])
    df['emotion'] = emotion
    emotions_df = pd.concat([emotions_df,df])
    
conn.close()

# pull in song title and artist
songs = pd.read_csv(
    'http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt', delimiter='<SEP>', header=None
)
songs.columns=['track_id','song_id','artist','song_title']

emotions_df = emotions_df.merge(songs, on='track_id')

# pickle
emotions_df.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df')


# grab subset of dataframe, get information from Spotify API for subset
emotions_df_subset = pd.DataFrame()

for emotion in emotions:
    emotion_subset = emotions_df[emotions_df['emotion'] == emotion][:1000]
    emotions_df_subset = pd.concat([emotions_df_subset,emotion_subset])
    
# collect features for songs in subset
song_features = {}
feature_list = [
    'duration','end_of_fade_in','start_of_fade_out','loudness','tempo','time_signature','key','mode'
    ]
client_id='8d91eae87c9c48c78b9f9cd08b9bc8e2',
client_secret='c1d699b329f24f74ab3d24fab21c0764'

for index in emotions_df_subset.index:
    print(index)
    try:
        artist = emotions_df_subset.loc[index]['artist']
        song = emotions_df_subset.loc[index]['song_title']
        track_id = emotions_df_subset.loc[index]['track_id']
    
        ind_song_analysis = audio_analysis(
            artist,song,feature_list, client_id=client_id, client_secret=client_secret
            )
    
        if ind_song_analysis != None:
            song_features[track_id] = ind_song_analysis
        
        sleep(0.5)
    except ReadTimeout:
        pass

for feature in feature_list:
    emotions_df_subset[feature] = np.NaN
    
for track_id in list(song_features.keys()):
    for feature in feature_list:
        emotions_df_subset.loc[emotions_df_subset['track_id']==track_id, feature] = song_features[track_id][feature]


emotions_df_subset.dropna(inplace=True)
emotions_df_subset.reset_index(inplace=True, drop=True)
emotions_df_subset.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df_subset')