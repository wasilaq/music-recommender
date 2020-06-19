I wanted to build a music recommender that places less emphasis on audio features, so that users get more variety and are more likely to discover something new and interesting. This is a music recommender that aims to reflect the user's thoughts and feelings. It's based on the content of songs and the emotions corresponding to songs.

## File Overview
Classes and functions used in these files can be viewed separately, in the *modules* folder.

### Topic Modeling
gather_lyrics_data.py  
topic_modeling.py

### Emotion Classification
gather_emotion_data.py  
emotion_modeling_setup.py

**Models**  
modeling_happy.py  
modeling_sad.py  
modeling_calm.py  
modeling_energetic.py

**Application**  
emotion_classification_utilization.py

### Recommendor
recommender.py
