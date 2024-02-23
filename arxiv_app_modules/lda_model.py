import pandas as pd
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['non','also'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


#simple string cleaner
def simple_cleaner(my_string):
    #remove punctuation
    my_string = my_string.translate(str.maketrans(" "," ",string.punctuation))
    
    #lower case, split, remove stopwords
    my_string = [wnl.lemmatize(w) for w in my_string.lower().split() if w not in stop_words]
    
    return " ".join(my_string)


def vectorizer(df):
	count_text_vectorizer = CountVectorizer(min_df=5, max_df=0.5)
	count_text_vectors = count_text_vectorizer.fit_transform(df['cleaned_text'])

	return count_text_vectorizer, count_text_vectors



#def lda_maker(count_text_vectors, count_text_vectorizer):
#	lda_text_model = LatentDirichletAllocation(n_components=6)
#	W_lda_text_matrix = lda_text_model.fit_transform(count_text_vectors)
#	H_lda_text_matrix = lda_text_model.components_
#	
#	lda_display = pyLDAvis.lda_model.prepare(lda_text_model, count_text_vectors,
#                                        count_text_vectorizer, sort_topics=False)
#
#	return lda_display

def lda_maker(count_text_vectors, count_text_vectorizer):
	lda_text_model = LatentDirichletAllocation(n_components=6, random_state=4)
	W_lda_text_matrix = lda_text_model.fit_transform(count_text_vectors)
	H_lda_text_matrix = lda_text_model.components_
	
	lda_display = pyLDAvis.lda_model.prepare(lda_text_model, count_text_vectors,
                                        count_text_vectorizer, sort_topics=False)

	return lda_display, W_lda_text_matrix, lda_text_model


def paper_output_maker(df):
	for paper in range(0,15):#len(df)):
		try:
			tab3.markdown(f"##### {df['Title'][paper]}") #print the title
			tab3.caption(f"Published {df['Published'][paper][:10]}") #published date
			tab3.markdown(f"**Summary**:  {df['Summary'][paper]}") # abstract / summary
			tab3.write(f"Link: {df['ID'][paper]}")
			tab3.divider()
		except: #if there are fewer than 15 papers it throws an error
			break


