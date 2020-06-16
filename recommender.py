#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:26:23 2020

@author: wasilaq
"""

from numpy import dot, array
from numpy.linalg import norm
import pandas as pd

# data
# topics
topic_columns = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10', 'topic11', 'topic12', 'topic13']
topics = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics_and_topics')[topic_columns]

# emotions
emotions = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/classification_modeling/emotions_df')[['track_id', 'emotion']]
emotions = emotions.set_index(emotions['track_id'])
emotions.drop_duplicates(inplace=True)
emotions = emotions.drop(columns=['track_id'])

emotions.loc[emotions['emotion'] == 'joyous','emotion'] = 'happy'
emotions.loc[emotions['emotion'] == 'relax','emotion'] = 'calm'
emotions.loc[emotions['emotion'] == 'soothing','emotion'] = 'calm'
emotions.loc[emotions['emotion'] == 'pain','emotion'] = 'sad'

cleaned_emotions = pd.get_dummies(emotions.reset_index(), columns=['emotion'])
cleaned_emotions = cleaned_emotions.groupby('track_id').sum()

for column in cleaned_emotions.columns:
    cleaned_emotions[column] = cleaned_emotions[column].apply(lambda x: 1 if x>0 else 0)


# merge topics and emotion data
topic_and_emotion = pd.merge(topics, cleaned_emotions, left_index=True, right_index=True)
topic_and_emotion.drop_duplicates(inplace=True)

# merge in songs that were assigned emotion tags by the classifiers
new_songs = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/new_songs')[['emotion_happy','emotion_sad','emotion_calm','emotion_energetic']]
cleaned_emotions_updated = pd.concat([cleaned_emotions,new_songs])
topic_and_emotion_updated = pd.merge(topics, cleaned_emotions_updated, left_index=True, right_index=True).drop_duplicates()

topic_and_emotion_updated.to_csv('/Users/wasilaq/Metis/music-recommender/pickled/topic_and_emotion_updated.csv')
# topic_and_emotion_updated = pd.read_csv('/Users/wasilaq/Metis/music-recommender/pickled/topic_and_emotion_updated.csv', index_col=0)


# grab metadata
songs = pd.read_csv(
    'http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt', delimiter='<SEP>', header=None
)
songs.columns=['track_id','song_id','artist','song_title']
songs.drop(columns=['song_id'], inplace=True)


# recommender
def recommendations(track_id, df=topic_and_emotion_updated, dist_type='cosine', num_recommendations=4):
    '''
    Returns recommended song based on input song.

    Parameters
    ----------
    track_id : str
        Input song, in the form of the track_id.
    df : DataFrame
        Dataset from which recommendations are being made. Index is track id.
    dist_type : str, optional
        How to measure distance between songs. Can be 'cosine' or 'euclidean'. The default is 'cosine'.
    num_recommendations : int, optional
        Number of recommendations in output. The default is 4.

    Returns
    -------
    recommendations : list
        Recommended songs (includes track id, song title, and artist for each recommendation).

    '''
    song_row = array(df.loc[track_id])
    distances = {}
    for new_track in df.index:
        row = array(df.loc[new_track])
        if dist_type=='euclidean':
            dist = norm(song_row-row)
        else:
            cos_sim = (dot(song_row, row))/(norm(song_row)*norm(row))
            dist = 1 - cos_sim
        distances[dist] = new_track
    
    shortest_distances = (
        sorted(distances.keys())[:(num_recommendations+1)]
        )
    most_similar_tracks = set()
    for distance in shortest_distances:
        most_similar_tracks.add(distances[distance])
    most_similar_tracks.remove(track_id)
    
    recommendations = [['Track ID', 'Song', 'Artist']]
    for track in most_similar_tracks:
        info = songs[songs['track_id']==track]
        song = info['song_title'].values[0]
        artist = info['artist'].values[0]
        recommendations.append([track, song, artist])
    
    return recommendations