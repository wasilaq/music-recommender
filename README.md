This is a music recommender that aims to reflect the user's thoughts and feelings. It's based on the content of songs and the emotions corresponding to songs and places less emphasis on the audio features so that users get more variety.

## File Directory
emotion_modeling_setup.py: format datasets for modeling  
functions/classification_functions: contains functions used in modeling for classifiers  
functions/spotify_api_functions: contains functions used to obtain data from Spotify's API  
gather_lyrics_data.py: obtain song lyrics
gather_emotion_data.py: obtain & clean dataset with emotion tags, obtain audio data (to use as features in classifiers)  
topic_modeling.py: topic modeling on lyrics  
models/modeling_calm.py: classifier designating is a song is "calm"  
models/modeling_energetic.py: classifier designating is a song is "energetic"  
models/modeling_happy.py: classifier designating is a song is "happy"  
models/modeling_sad.py: classifier designating is a song is "sad"  
predict_emotions.py: apply classifiers to new data  
recommender.py: code for recommendation system  
