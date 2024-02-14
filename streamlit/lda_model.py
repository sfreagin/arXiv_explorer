import pandas as pd
import string

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

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



def lda_maker(count_text_vectors, count_text_vectorizer):
	lda_text_model = LatentDirichletAllocation(n_components=6)
	W_lda_text_matrix = lda_text_model.fit_transform(count_text_vectors)
	H_lda_text_matrix = lda_text_model.components_
	
	lda_display = pyLDAvis.lda_model.prepare(lda_text_model, count_text_vectors,
                                        count_text_vectorizer, sort_topics=False)

	return lda_display