#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:53:39 2020

@author: wasilaq
"""

def audio_analysis(artist, song, client_id, client_secret, features=['duration', 'sample_md5', 'offset_seconds', 'window_seconds', 'analysis_sample_rate', 'analysis_channels', 'end_of_fade_in', 'start_of_fade_out', 'loudness', 'tempo', 'tempo_confidence', 'time_signature','time_signature_confidence', 'key', 'key_confidence', 'mode', 'mode_confidence']):
    '''
    

    Parameters
    ----------
    artist : TYPE
        DESCRIPTION.
    song : TYPE
        DESCRIPTION.
    sp : TYPE
        DESCRIPTION.
    features : TYPE
        DESCRIPTION.

    Returns
    -------
    ind_song_analysis : TYPE
        DESCRIPTION.

    '''
    credentials = oauth2.SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
        )

    token = credentials.get_access_token()
    sp = spotipy.Spotify(auth=token)

    try:
        sp_tracks = sp.search(q='artist:' + artist + ' track:' + song, type='track')
    except spotipy.client.SpotifyException:
        token = credentials.get_access_token()
        sp = spotipy.Spotify(auth=token)
        sp_tracks = sp.search(q='artist:' + artist + ' track:' + song, type='track')

    try:
        sp_track_id = sp_tracks['tracks']['items'][0]['id']
        try:
            sp_analysis = sp.audio_analysis(sp_track_id)
        except spotipy.client.SpotifyException:
            token = credentials.get_access_token()
            sp = spotipy.Spotify(auth=token)
            sp_analysis = sp.audio_analysis(sp_track_id)
        ind_song_analysis = {}
        for feature in feature_list:
            ind_song_analysis[feature] = sp_analysis['track'][feature]
        return ind_song_analysis
            
    except IndexError:
        pass