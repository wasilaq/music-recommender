I wanted to build a music recommender that places less emphasis on audio features, so that users get more variety and are more likely to discover something new and interesting. This is a music recommender that aims to reflect the user's thoughts and feelings. It's based on the content of songs and the emotions corresponding to songs.

## File Overview
Custom functions used in these files can be found in the *functions* folder.

### Obtain Data
gather_lyrics_data.py  
gather_emotion_data.py  

### Modeling
topic_modeling.py
emotion_modeling_setup.py  
***Models*** **Folder**  
*Contains classification models*  
modeling_happy.py  
modeling_sad.py  
modeling_calm.py  
modeling_energetic.py

### Modeling Application
predict_emotions.py

### Recommendor
recommender.py
