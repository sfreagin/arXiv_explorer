import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

import os
try:
	API_KEY = st.secrets['API_KEY']
except:
	from arxiv_app_modules.config import API_KEY

import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('non')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.lda_model

from bokeh.plotting import figure
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from wordcloud import WordCloud
from arxiv_app_modules.wordcloud_generator import generate_wordcloud_from_df

from arxiv_app_modules.arxiv_api import arxiv_query, download_pdf_from_link, extract_text_from_pdf
from arxiv_app_modules.lda_model import simple_cleaner, vectorizer, lda_maker
from arxiv_app_modules.summarization import generate_summary, batch_input_text, query


################################################################
################################################################
################################################################
################################################################

######## statefulness is recommended for streamlit apps ########
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

def session_reset():
    st.session_state.clicked = False


######## INTRO PAGE ########
st.title("arXiv.org Explorer")
st.write("This app provides a summary of arXiv.org preprint activity in the subdomain of your choice. \
	Thank you to arXiv for use of its open access interoperability. https://arxiv.org")
st.caption("For avoidance of doubt, we are independent researchers and this project is NOT affiliated with \
	the arXiv. This app was not reviewed or approved by, \
	nor does it necessarily express or reflect the policies or opinions of, arXiv.")

#creating tabs on streamlit app
tab1, tab2, tab4, tab3, tab5 = st.tabs(['Home Inputs', 'Simple Stats', 'LDA Analysis', 'Notable Papers', 
	 'Summarizer (BETA)'])

################################################################
#### USER SELECTION OF CATEGORIES ##############################
################################################################

field_choice = tab1.radio(
	"Please select a field:",
	('Computer Science', 'Economics','Electrical Engineering and Systems Science', 
		'Math', 'Physics','Quantitative Biology', 'Quantitative Finance', 'Statistics',),
	on_change = session_reset)

#pull in the categories corresponding to each field
with open('category_taxonomy.json','r') as f:
	category_dict = json.load(f)

#select box for category choice
category_choice = tab1.selectbox(
	f"Choose a {field_choice} category:", category_dict[field_choice].keys(),
	on_change = session_reset)

#write out the user's choice
subcategory = " ".join(category_choice.split()[:-1])
tab1.write(f"You have chosen the {field_choice} field: {subcategory}")

#choose a date range
day_dict = {'7 days':7, '30 days':30, '90 days':90}
day_choice = tab1.radio("**Choose a date window**",day_dict.keys(), on_change = session_reset)
date_choice = tab1.date_input("Choose the endpoint", format="YYYY-MM-DD")
tab1.write(f"Query from {date_choice-timedelta(days=day_dict[day_choice])} to {date_choice}")

#these date variables will go into the API call
start_date = "".join(str(date_choice-timedelta(days=day_dict[day_choice])).split('-'))
end_date = "".join(str(date_choice).split('-'))

#run the API query
arxiv_button = tab1.button("Pull arXiv.org info", on_click=click_button)
st.divider()


################################################################
#### ARXIV API QUERY ###########################################
################################################################
#### this pulls data from arxiv.org which powers the whole script
################################################################

chosen_category = category_dict[field_choice][category_choice]
if chosen_category == "None":
	st.session_state.clicked = False

if st.session_state.clicked:
	df = arxiv_query(chosen_category,start_date,end_date)
	df = df.sort_values(by='Published', ascending=False).reset_index(drop=True)

	if len(df) == 1000:
		tab1.write(f"##### We stopped after finding {len(df)} papers - \
			consider narrowing the date range of your search")
	elif len(df) <= 10:
		tab1.write(f"##### There are only {len(df)} papers in this date range - \
			consider expanding the date range of your search")
	else:
		tab1.write(f"##### There are {len(df)} papers in this date range - \
			explore the tabs for more info")
else: #example output so the app doesn't crash upon opening
	df = pd.read_csv('arxiv_app_modules/data/example_output.csv')


################################################################
#### LDA MODEL #################################################
################################################################
#### uses sklearn and pyLDAvis for topic modeling
################################################################

tab4.header(f"Latent Dirichlet Allocation (LDA) Analysis - {subcategory}")
tab4.markdown('Latent Dirichlet Allocation (LDA) is an unsupervised machine learning method to organize \
	text documents by "topic" according to their vocabulary. Even narrow scientific fields will often span \
	multiple overlapping topics, and using LDA models can help a researcher narrow the scope of papers \
	relevant to their topic of interest.')
tab4.markdown('The LDA visual model below is powered by the `pyLDAvis` library \
	and is designed to ["help users interpret the topics in... \
	a corpus of text data."](https://pyldavis.readthedocs.io/en/latest/readme.html) ')
tab4.divider()

#clean text, run custom sklearn model, output pyLDAvis interactive chart
if st.session_state.clicked:
	if len(df) <= 10:
		tab4.subheader(":red[WARNING: not enough papers to create LDA visual]")
	else:
		#light cleaning
		df['Summary'] = df['Summary'].str.replace('-', ' ')
		df['cleaned_text'] = df['Summary'].apply(simple_cleaner)	

		#create the lda_display inputs
		count_text_vectorizer, count_text_vectors = vectorizer(df)
		lda_display, W_lda_text_matrix, lda_text_model = lda_maker(count_text_vectors, count_text_vectorizer)

		#display the pyLDAvis output
		with tab4:
			html_string = pyLDAvis.prepared_data_to_html(lda_display)
			st.components.v1.html(html_string, width=1500, height=800, scrolling=True)

tab4.subheader('Looking under the hood')
tab4.markdown("""This LDA model is based on the scikit-learn `LatentDirichletAllocation` library. \
	The sklearn code is designed for Singular Value Decomposition (SVD) and Non-Negative Matrix \
	Factorization (NMF), both of which use matrix factorization for topic modeling, \
	but it outputs similar array shapes for LDA as well.

For more information, please visit the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html).

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis

#### turn the text strings into token vectors
#### but only if a token appears in at least 5 and no more than 50% of documents 
count_text_vectorizer = CountVectorizer(min_df=5, max_df=0.5)
count_text_vectors = count_text_vectorizer.fit_transform(df['cleaned_text'])

#### creates LDA analogs for the W (topic-document) and H (topic-feature) matrices of SVD and NMF
lda_text_model = LatentDirichletAllocation(n_components=6, random_state=4)
W_lda_text_matrix = lda_text_model.fit_transform(count_text_vectors)
H_lda_text_matrix = lda_text_model.components_

#### pyLDAvis provides an easy visual library
lda_display = pyLDAvis.lda_model.prepare(lda_text_model, count_text_vectors,
                                        count_text_vectorizer, sort_topics=False)

```""")


################################################################
#### NOTABLE PAPER OUTPUTS  ####################################
################################################################
#### prints out article abstracts and sorts by LDA topic
################################################################

# this function iteratively writes paper information: date, abstract, link, etc.
def paper_output_maker(df):
	for paper in range(0,20):#len(df)):
		try:
			tab3.markdown(f"##### {df['Title'][paper]}") #print the title
			tab3.caption(f"Topic relevance: {round(df['score'][paper]*100,1	)}% $\cdot$ \
							Published {df['Published'][paper][:10]}") #published date
			tab3.markdown(f"**Abstract**:  {df['Summary'][paper]}") #abstract (summary)
			tab3.write(f"Link: {df['ID'][paper]}") #URL
			tab3.divider()
		except: #if there are fewer than 20 papers it throws an error
			break


tab3.header(f"Notable Papers in {subcategory}")
tab3.write(f"**{date_choice-timedelta(days=day_dict[day_choice])}** $\longleftrightarrow$ **{date_choice}**")

#this creates the menu of LDA topics to sort by
if st.session_state.clicked:
	if len(df) <= 10:
		tab3.subheader(":red[WARNING: not enough papers to sort by topic]")
	else:
		#choose the LDA topic
		features = count_text_vectorizer.get_feature_names_out()
		topic_extender = [f"Topic {topic+1} keywords: {[features[words.argsort()[::-1][i]] for i in range(0,6)]}" for topic, words in enumerate(lda_text_model.components_)]

		topic = tab3.radio("Order by:",
			["Chronological order"] + topic_extender)

		topic_mapping = {"Chronological order": None, 
							"Topic 1": 0, 
							"Topic 2": 1, 
							"Topic 3": 2, 
							"Topic 4": 3,
							"Topic 5": 4,
							"Topic 6": 5}

tab3.caption("\*Topic keywords are based on the unsupervised LDA topics under `LDA Analysis`")
tab3.divider()

#write the abstracts based on sorting criteria
if st.session_state.clicked:
	#light string editing
	df['Summary'] = df['Summary'].str.replace('\n',' ') # removing line breaks
	df['Title'] = df['Title'].str.replace('\n','') # removing line breaks

	if len(df) <= 10:
		df['score'] = 1
		paper_output_maker(df)
	else:		
		if topic == "Chronological order":
			df['score'] = 1
			paper_output_maker(df)
		else:
			df['score'] = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic[:7]]]
			ranked_list = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic[:7]]].sort_values(ascending=False).index
			paper_output_maker(df.iloc[ranked_list].reset_index())


################################################################
#### SIMPLE STATS ##############################################
################################################################
#### tokens, charts, wordcloud, etc.
################################################################

tab2.header(f"arXiv.org Abstracts in {subcategory}")
tab2.write(f"**{date_choice-timedelta(days=day_dict[day_choice])}** $\longleftrightarrow$ **{date_choice}**")

######## NUMERICAL STATS ########
number_of_papers = len(df)
summary_lengths = df['Summary'].str.split().map(len)

col1, col2, col3, col4 = tab2.columns(4)
col1.metric("Number of Papers", f"{number_of_papers}")
col2.metric("Avg. Abstract Length", f'{int(summary_lengths.mean())} words')
col3.metric("Total Abstract Words", f"{len(np.concatenate(df['Summary'].str.split()))}")
col4.metric("Unique Abstract Words", f"{len(set(np.concatenate(df['Summary'].str.split())))}")
tab2.divider()

######## PLOTTING CHARTS ########

#### BOKEH histogram
hist, edges = np.histogram(summary_lengths, density=False, bins=int(np.sqrt(len(df))))
p = figure(title=f"Abstract Length in Words (tokens)",
            x_axis_label="Abstract Length (tokens)", width=600, height=300)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
hover = HoverTool(tooltips = [('Papers', "@top")])
p.add_tools(hover)

#### BOKEH bar chart by date
sorted_dates = pd.to_datetime(df['Published']).dt.date.sort_values()
dates_df = pd.DataFrame(sorted_dates.value_counts()).sort_values(by='Published').reset_index()
source = ColumnDataSource(dates_df)
p_bar = figure(title="Pre-print Submissions by Date",x_axis_type='datetime', width=700, height=350)
p_bar.vbar(source=source, x='Published', top='count',width=80000000,line_color='white')
hover = HoverTool(tooltips = [ ("Date","@Published{%F}"),('Papers', '@count')],
                  formatters={'@Published': 'datetime'})
p_bar.add_tools(hover)

######## DISPLAYING ALL CHARTS PLUS WORDCLOUD ########
with tab2:
	st.bokeh_chart(p,use_container_width=False)
	st.divider()
	st.bokeh_chart(p_bar,use_container_width=False)
	st.divider()
	generate_wordcloud_from_df(df, field_choice, subcategory)
	if st.session_state.clicked:
		st.image('wordcloud_output.png')


################################################################
#### HUGGINGFACE MODEL TO SUMMARIZE ARTICLES ###################
################################################################

# It has been said that astronomy is a humbling and character-building experience. 
# There is perhaps no better demonstration of the folly of human conceits than this 
# distant image of our tiny world. To me, it underscores our responsibility 
# to deal more kindly with one another, and to preserve and cherish the pale blue dot, 
# the only home we've ever known.
#
# — Carl Sagan, Pale Blue Dot, 1994

################################################################
#### SUMMARIZER LLM  ###########################################
################################################################
#### powers the huggingface model to summarize an article PDF
################################################################

tab5.write("[Provide credit to] https://huggingface.co/philschmid/bart-large-cnn-samsum")

link = tab5.text_input("Paste link to arXiv.org paper or PDF link")
model_button = tab5.button("Summarize article")
tab5.caption('Note: this model in still in beta mode and may stop for unknown reasons. \
	If it does not work, wait 15-30 seconds and hit the "Summarize article" button again.')

with tab5:
	if model_button:
		link = link.replace('abs', 'pdf')  # Replacing 'abs' with 'pdf' in the URL
		if not link.endswith('.pdf'):  # Checking if the link ends with '.pdf'
		    link += '.pdf'  # Appending '.pdf' to the link if it doesn't end with it already

		pdf_io = download_pdf_from_link(link) # download the PDF
		pdf_text = extract_text_from_pdf(pdf_io) # extract text

		article_summary = generate_summary(input_text=pdf_text, API_KEY=API_KEY) #call the external LLM
		st.markdown(f"##### Main idea: {article_summary}")


####################


