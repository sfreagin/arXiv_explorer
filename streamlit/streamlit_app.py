import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta

from arxiv_api import arxiv_query

import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('non')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.lda_model

from lda_model import simple_cleaner, vectorizer, lda_maker#, paper_output_maker


#statefulness
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True


# INTRO PAGE
st.title("arXiv.org Summarizer")
st.write("This app provides a summary of arXiv.org preprint activity in the subdomain of your choice")

tab1, tab2, tab4, tab3 = st.tabs(['User Input', 'Summary', 'LDA Analysis', 'Papers'])

################################################################
#### USER SELECTION OF CATEGORIES ##############################
################################################################

field_choice = tab1.radio(
	"Please select a field:",
	('Math','Physics'))

category_dict = {
	"Math" : { "None": "none",
				"Algebraic Geometry (math.AG)": "math.AG",
				"Algebraic Topology (math.AT)": "math.AT",
				"Analysis of PDEs (math.AP)": "math.AP",
				"Category Theory (math.CT)": "math.CT",
				"Complex Variables (math.CV)": "math.CV",
				"Classical Analysis and ODEs (math.CA)": "math.CA",
				"Combinatorics (math.CO)": "math.CO",
				"Commutative Algebra (math.AC)": "math.AC",
				"Differential Geometry (math.DG)": "math.DG",
				"Dynamical Systems (math.DS)": "math.DS",
				"Functional Analysis (math.FA)": "math.FA",
				"General Mathematics (math.GM)": "math.GM",
				"General Topology (math.GN)": "math.GN",
				"Geometric Topology (math.GT)": "math.GT",
				"Group Theory (math.GR)": "math.GR",
				"History and Overview (math.HO)": "math.HO",
				"Information Theory (math.IT)": "math.IT",
				'K-Theory and Homology (math.KT)': "math.KT",
				"Logic (math.LO)": "math.LO",
				"Mathematical Physics (math.MP)": "math.MP",
				"Metric Geometry (math.MG)": "math.MG",
				"Number Theory (math.NT)": "math.NT",
				"Numerical Analysis (math.NA)": "math.NA",
				"Operator Algebras (math.OA)": "math.OA",
				"Optimization and Control (math.OC)": "math.OC",
				"Probability (math.PR)": "math.PR",
				"Quantum Algebra (math.QA)": "math.QA",
				"Representation Theory (math.RT)": "math.RT",
				"Rings and Algebras (math.RA)": "math.RA",
				"Spectral Theory (math.SG)": "math.SG",
				"Statistics Theory (math.ST)": 'math.ST',
				"Symplectic Geometry (math.SG)": "math.SG",
				},
	"Physics" : { "None ": "none",
				#astrophysics
				"ASTROPHYSICS -  Astrophysics of Galaxies (astro-ph.GA)": "astro-ph.GA",
				"ASTROPHYSICS - Cosmology and Nongalactic Astrophysics (astro-ph.CO)": "astro-ph.CO",
				"ASTROPHYSICS - Earth and Planetary Astrophysics (astro-ph.EP)": "astro-ph.EP",
				"ASTROPHYSICS - High Energy Astrophysical Phenomena (astro-ph.HE)": "astro-ph.HE",
				"ASTROPHYSICS - Instrumentation and Methods for Astrophysics (astro-ph.IM)": 'astro-ph.IM',
				"ASTROPHYSICS - Mathematical Physics (math-PH)": "math-PH",
				"ASTROPHYSICS - Solar and Stellar Astrophysics (astro-ph.SR)": "astro-ph.SR",
				#condensed matter
				"CONDENSED MATTER - Disordered Systems and Neural Networks (cond-mat.dis-nn)": "cond-mat.dis-nn",
				"CONDENSED MATTER - Materials Science (cond-mat.mtrl-sci)": "cond-mat.mtrl-sci",
				"CONDENSED MATTER - Mesoscale and Nanoscale Physics {cond-mat.mes-hall}": "cond-mat.mes-hall",
				"CONDENSED MATTER - Other Condensed Matter (cond-mat.other)": "cond-mat.other",
				"CONDENSED MATTER - Quantum Gases (cond-mat.quant-gas)": "cond-mat.quant-gas",
				"CONDENSED MATTER - Soft Condensed Matter (cond-mat.soft)": "cond-mat.soft",
				"CONDENSED MATTER - Statistical Mechanics (cond-mat.stat-mech)": "cond-mat.stat-mech",
				"CONDENSED MATTER - Strongly Correlated Electrons (cond-mat.str-el)": "cond-mat.str-el",
				"CONDENSED MATTER - Superconductivity (cond-mat.supr-con)": "cond-mat.supr-con",
				# General Relativity
				"GENERAL RELATIVITY & Quantum Cosmology (gr-qc)": "gr-qc",
				# high energy
				"HIGH ENERGY PHYSICS - Experiment (hep-ex)": "hep-ex",
				"HIGH ENERGY PHYSICS - Lattice (hep-lat)": "hep-lat",
				"HIGH ENERGY PHYSICS - Phenomenology (hep-ph)": "hep-ph",
				"HIGH ENERGY PHYSICS - Theory (hep-th)": "hep-th",
				#mathematical physics
				"MATHEMATICAL PHYSICS (math-ph)": "math-ph",
				#nonlinear sciences
				"NONLINEAR SCIENCES - Adaptation and Self-Organizing Systems (nlin.AO)": "nlin.AO",
				"NONLINEAR SCIENCES - Cellular Automata and Lattice Gases (nlin.CG)": "nlin.CG",
				"NONLINEAR SCIENCES - Chaotic Dynamics (nlin.CD)": "nlin.CD",
				"NONLINEAR SCIENCES - Exactly Solvable and Integrable Systems (nlin.SI)": "nlin.SI",
				"NONLINEAR SCIENCES - Pattern Formation and Solitons (nlin.PS)": "nlin.PS",
				#nuclear
				"NUCLEAR EXPERIMENT (nucl-ex)": "nucl-ex",
				"NUCLEAR THEORY (nucl-th)": "nucl-th",
				# Physics
				"PHYSICS - Accelerator Physics (physics.acc-ph)": "physics.acc-ph",
				"PHYSICS - Applied Physics (physics.app-ph)": "physics.app-ph",
				"PHYSICS - Atmospheric and Oceanic Physics (physics.ao-ph)": "physics.ao-ph",
				"PHYSICS - Atomic and Molecular Clusters (physics.atm-clus)": "physics.atm-clus",
				"PHYSICS - Atomic Physics (physics.atom-ph)": "physics.atom-ph",
				"PHYSICS - Biological Physics (physics.bio-ph)": "physics.bio-ph",
				"PHYSICS - Chemical Physics (physics.chem-ph)": "physics.chem-ph",
				"PHYSICS - Classical Physics (physics.class-ph)": "physics.class-ph",
				"PHYSICS - Computational Physics (physics.comp-ph)": "physics.comp-ph",
				"PHYSICS - Data Analysis, Statistics and Probability (physics.data-an)": "physics.data-an",
				"PHYSICS - Fluid Dynamics (physics.flu-dyn)": "physics.flu-dyn",
				"PHYSICS - General Physics (physics.gen-ph)": "physics.gen-ph",
				"PHYSICS - Geophysics (physics.geo-ph)": "physics.geo-ph",
				"PHYSICS - History and Philosophy of Physics (physics.hist-ph)": "physics.hist-ph",
				"PHYSICS - Instrumentation and Detectors (physics.ins-det)": "physics.ins-det",
				"PHYSICS - Medical Physics (physics.med-ph)": "physics.med-ph",
				"PHYSICS - Optics (physics.optics)": "physics.optics",
				"PHYSICS - Physics and Society (physics.soc-ph)": "physics.soc-ph",
				"PHYSICS - Physics Education (physics.ed-ph)": "physics.ed-ph",
				"PHYSICS - Plasma Physics (physics.plasm-ph)": "physics.plasm-ph",
				"PHYSICS - Popular Physics (physics.pop-ph)": "physics.pop-ph",
				"PHYSICS - Space Physics (physics.space-ph)": "physics.space-ph",
				#quantum
				"QUANTUM PHYSICS (quant-ph)": "quant-ph"
				}
}

#select box for category choice
category_choice = tab1.selectbox(
	f"Choose a {field_choice} category:", category_dict[field_choice].keys())

#write out the user's choice
subcategory = " ".join(category_choice.split()[:-1])
tab1.write(f"You have chosen the {field_choice} field: {subcategory}")

#choose a date range
tab1.markdown('**Choose a date range**')
day_dict = {'7 days':7, '30 days':30, '90 days':90}
day_choice = tab1.radio("Number of Days",day_dict.keys())
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
	df = pd.read_csv('data/example_output.csv')
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

if st.session_state.clicked:
	#light cleaning
	df['Summary'] = df['Summary'].str.replace('-', ' ')
	df['cleaned_text'] = df['Summary'].apply(simple_cleaner)	

	#create the lda_display inputs
	count_text_vectorizer, count_text_vectors = vectorizer(df)
	lda_display, W_lda_text_matrix = lda_maker(count_text_vectors, count_text_vectorizer)

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
			tab3.caption(f"Topic relevance: {round(df['score'][paper]*100,0)}% $\cdot$ \
							Published {df['Published'][paper][:10]}") #published date
			tab3.markdown(f"**Summary**:  {df['Summary'][paper]}") # abstract / summary
			tab3.write(f"Link: {df['ID'][paper]}")
			tab3.divider()
		except: #if there are fewer than 15 papers it throws an error
			break

#header
tab3.header(f"Notable Papers in {subcategory}")

#choose the LDA topic
topic = tab3.radio("Order by:",
	["Chronological", "Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6"])

topic_mapping = {"Chronological": None, 
					"Topic 1": 0, 
					"Topic 2": 1, 
					"Topic 3": 2, 
					"Topic 4": 3,
					"Topic 5": 4,
					"Topic 6": 5}

if st.session_state.clicked:
	tab3.divider()
	#light string editing
	df['Summary'] = df['Summary'].str.replace('\n',' ') # removing line breaks
	df['Title'] = df['Title'].str.replace('\n','') # removing line breaks

	if topic_mapping[topic] == None:
		df['score'] = 1
		paper_output_maker(df)
	else:
		df['score'] = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic]]
		ranked_list = pd.DataFrame(W_lda_text_matrix)[topic_mapping[topic]].sort_values(ascending=False).index
		paper_output_maker(df.iloc[ranked_list].reset_index())




####################


