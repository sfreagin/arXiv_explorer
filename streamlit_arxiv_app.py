import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

from arxiv_app_modules.arxiv_api import arxiv_query, download_pdf_from_link, extract_text_from_pdf
from arxiv_app_modules.summarization import generate_summary, batch_input_text, query

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

from arxiv_app_modules.lda_model import simple_cleaner, vectorizer, lda_maker#, paper_output_maker


#statefulness
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

def session_reset():
    st.session_state.clicked = False

# INTRO PAGE
st.title("arXiv.org Explorer")
st.write("This app provides a summary of arXiv.org preprint activity in the subdomain of your choice")

tab1, tab3, tab4, tab2, tab5 = st.tabs(['User Input',  'Notable Papers', 'LDA Analysis', 
	'Statistics', 'Summarizer (BETA)'])

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
#tab1.markdown('**Choose a date range**')
day_dict = {'7 days':7, '30 days':30, '90 days':90}
day_choice = tab1.radio("**Choose a date window**",day_dict.keys(), on_change = session_reset)
date_choice = tab1.date_input("Choose a starting date", format="YYYY-MM-DD")
tab1.write(f"Query from {date_choice-timedelta(days=day_dict[day_choice])} to {date_choice}")

#these date variables will go into the API call
start_date = "".join(str(date_choice-timedelta(days=day_dict[day_choice])).split('-'))
end_date = "".join(str(date_choice).split('-'))

#run the API query
arxiv_button = tab1.button("Pull arXiv.org info", on_click=click_button)
st.divider()


################################################################
#### ARXIV API QUERY        ####################################
################################################################

chosen_category = category_dict[field_choice][category_choice]
if chosen_category == "None":
	st.session_state.clicked = False

if st.session_state.clicked:
	df = arxiv_query(chosen_category,start_date,end_date)
	df = df.sort_values(by='Published', ascending=False).reset_index(drop=True)
	#df.to_pickle('temp_df.pkl')
	if len(df) == 1000:
		tab1.write(f"##### We stopped after finding {len(df)} papers - \
			consider narrowing the date range of your search")
	else:
		tab1.write(f"##### There are {len(df)} papers in this date range - \
			click the Summary tab for more info")
else:
	df = pd.read_csv('arxiv_app_modules/data/example_output.csv')
################################################################
#### HUGGINGFACE MODEL TO SUMMARIZE ARTICLES ###################
################################################################

summary_text = """It has been said that astronomy is a humbling and character-building experience. \
There is perhaps no better demonstration of the folly of human conceits than this distant image of our tiny world. \
To me, it underscores our responsibility to deal more kindly with one another, and to preserve and cherish the pale blue dot, \
the only home we've ever known.

— Carl Sagan, Pale Blue Dot, 1994
"""

################################################################
#### SUMMARY OUTPUTS        ###############################
################################################################

tab2.header(f"arXiv.org - {subcategory}")
tab2.write(f"**{date_choice-timedelta(days=day_dict[day_choice])}** $\longleftrightarrow$ **{date_choice}**")

number_of_papers = len(df)
summary_lengths = df['Summary'].str.split().map(len)

col1, col2 = tab2.columns(2)
col1.metric("Number of Papers", f"{number_of_papers}")
col2.metric("Avg. Summary", f'{int(summary_lengths.mean())} words')

#### SUMMARY LENGTH HISTOGRAM
from bokeh.plotting import figure
from bokeh.io import show, output_file

hist, edges = np.histogram(summary_lengths, density=True, bins=32)
p = figure(title=f"Lengths of Summary Abstracts",
            x_axis_label="Tokens",width=100, height=300)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

tab2.bokeh_chart(p,use_container_width=True)

tab2.caption('NOTE TO STEPHEN: CREATE A WORDCLOUD')


################################################################
#### LDA MODEL        ###############################
################################################################

tab4.header(f"Latent Dirichlet Allocation (LDA) Analysis - {subcategory}")
tab4.caption("[NOTE TO STEPHEN: SHORT HOW-TO GUIDE GOES HERE]")
if st.session_state.clicked:
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

################################################################
#### SUMMARY (ABSTRACT) OUTPUTS  ###############################
################################################################

def paper_output_maker(df):
	for paper in range(0,20):#len(df)):
		try:
			tab3.markdown(f"##### {df['Title'][paper]}") #print the title
			tab3.caption(f"Topic relevance: {round(df['score'][paper]*100,1	)}% $\cdot$ \
							Published {df['Published'][paper][:10]}") #published date
			tab3.markdown(f"**Abstract**:  {df['Summary'][paper]}") # abstract / summary
			tab3.write(f"Link: {df['ID'][paper]}")
			tab3.divider()
		except: #if there are fewer than 15 papers it throws an error
			break

#header
tab3.header(f"Notable Papers in {subcategory}")
tab3.write(f"**{date_choice-timedelta(days=day_dict[day_choice])}** $\longleftrightarrow$ **{date_choice}**")

if st.session_state.clicked:
	#choose the LDA topic
	features = count_text_vectorizer.get_feature_names_out()
	topic_extender = [f"Topic {topic+1} keywords: {[features[words.argsort()[::-1][i]] for i in range(0,6)]}" for topic, words in enumerate(lda_text_model.components_)]

	topic = tab3.radio("Order by:",
		["Chronological"] + topic_extender)
	#	["Chronological", "Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6"])

	topic_mapping = {"Chronological": None, 
						"Topic 1": 0, 
						"Topic 2": 1, 
						"Topic 3": 2, 
						"Topic 4": 3,
						"Topic 5": 4,
						"Topic 6": 5}

tab3.divider()

if st.session_state.clicked:	
	#light string editing
	df['Summary'] = df['Summary'].str.replace('\n',' ') # removing line breaks
	df['Title'] = df['Title'].str.replace('\n','') # removing line breaks

	if topic == "Chronological":
		df['score'] = 1
		paper_output_maker(df)
	else:
		df['score'] = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic[:7]]]
		ranked_list = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic[:7]]].sort_values(ascending=False).index
		paper_output_maker(df.iloc[ranked_list].reset_index())





################################################################
#### PULL PDF CONTENT  ###############################
################################################################

tab5.write("[Provide credit to] https://huggingface.co/philschmid/bart-large-cnn-samsum")

link = tab5.text_input("Link to arXiv.org paper")
model_button = tab5.button("Summarize article")

#tab5.subheader('THIS DOES NOT WORK YET ONLINE')

if model_button:
	link = link.replace('abs', 'pdf')  # Replacing 'abs' with 'pdf' in the URL
	if not link.endswith('.pdf'):  # Checking if the link ends with '.pdf'
	    link += '.pdf'  # Appending '.pdf' to the link if it doesn't end with it already

	pdf_io = download_pdf_from_link(link)
	pdf_text = extract_text_from_pdf(pdf_io)

	article_summary = generate_summary(pdf_text)
	st.markdown(f"##### Main idea: {article_summary}")


####################


