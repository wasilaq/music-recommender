#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:10:42 2020

@author: wasilaq
"""
from time import sleep
from requests.exceptions import ReadTimeout
import pickle
from random import sample
from spotify_api_functions import audio_analysis
import pandas as pd

topics = (pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics_and_topics')).index

emotions = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df')['track_id']

no_emotion_tag = [x for x in list(topics) if x not in list(emotions)]

# test - no_emotion_tag should include songs from topics df that are not in emotions df
emotions.isin(['TRAAAAV128F421A322']).any() # false
'TRAAAAV128F421A322' in topics # true

no_tag_subset = sample(no_emotion_tag, 1000)


songs = pd.read_csv(
    'http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt', delimiter='<SEP>', header=None
)
songs.columns=['track_id','song_id','artist','song_title']
songs.drop(columns=['song_id'], inplace=True)

no_tag_songs = songs[songs['track_id'].isin(no_tag_subset)]
no_tag_songs.reset_index(inplace=True)


# Spotify API  
song_features = {}
feature_list = [
    'duration','end_of_fade_in','start_of_fade_out','loudness','tempo','time_signature','key','mode'
    ]
client_id='8d91eae87c9c48c78b9f9cd08b9bc8e2',
client_secret='c1d699b329f24f74ab3d24fab21c0764'

for index in no_tag_songs.index[168:]:
    print(index)
    try:
        artist = no_tag_songs.loc[index]['artist']
        song = no_tag_songs.loc[index]['song_title']
        track_id = no_tag_songs.loc[index]['track_id']
    
        ind_song_analysis = audio_analysis(
            artist, song, features=feature_list, client_id=client_id, client_secret=client_secret
            )
    
        if ind_song_analysis != None:
            song_features[track_id] = ind_song_analysis
        
        sleep(0.5)
    except ReadTimeout:
        pass
    
new_songs = pd.DataFrame(song_features.values(), index=song_features.keys())

models = ['happy','sad','calm','energetic']
fitted_models = {}
features = ['duration', 'end_of_fade_in', 'start_of_fade_out', 'loudness', 'tempo', 'time_signature', 'key', 'mode']

for model in models:
    fitted_models[model] = pickle.load(open('/Users/wasilaq/Metis/music-recommender/pickled/models/model_' + model, 'rb'))
    pred = fitted_models[model].predict(new_songs[features])
    new_songs[('emotion_'+model)] = pred.astype(int)


new_songs.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/new_songs')