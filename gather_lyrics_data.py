#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:34:26 2020

@author: wasilaq
"""
import pandas as pd
from sqlite3 import connect
from langid import classify


cnx = connect('/Users/wasilaq/Downloads/lyrics_dataset.db')
lyrics = pd.read_sql_query(
    "SELECT track_id,word,count FROM lyrics ORDER BY track_id LIMIT 5000000", cnx
    )
# column 'track_id' -> as usual, track id from the MSD
# column 'mxm_tid' -> track ID from musiXmatch
# column 'word' -> a word that is also in the 'words' table
# column 'count' -> word count for the word
# column 'is_test' -> 0 if this example is from the train set, 1 if test

cnx.close()

songs = pd.read_csv(
    'http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt', delimiter='<SEP>', header=None
)
songs.columns=['track_id','song_id','artist','song_title']

lyrics = lyrics.merge(songs, on='track_id')

# delete last track from bow - may not have grabbed full lyrics for this track from table
lyrics.tail()
lyrics = lyrics[lyrics['track_id'] != 'TRGUJQG128E078E718']


# convert to bag of words
bow = lyrics.pivot_table('count','track_id','word')
bow = bow.fillna(0)

# remove non-English songs
eng_bow = bow
for word in bow.columns:
    lang = classify(word)[0]
    if lang != 'en':
        eng_bow = eng_bow.drop(word, axis=1)
                
eng_bow['total'] = eng_bow.sum(axis=1)
eng_bow = eng_bow[eng_bow['total'] > 20]

lyrics.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics')
eng_bow.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics_bow')