import requests
import pandas as pd
from bs4 import BeautifulSoup


def arxiv_query(chosen_category, start_date, end_date):

	max_results=1000
	start_date=start_date
	end_date=end_date
	chosen_category=chosen_category

	api_url = f"http://export.arxiv.org/api/query?search_query=cat:{chosen_category}+AND+submittedDate:[{start_date}+TO+{end_date}]"

	params = {
	        "start": 0,  # Start index of results
	        "max_results": max_results,  # Maximum number of results to retrieve
	    }


	response = requests.get(api_url, params=params)
	soup = BeautifulSoup(response.content, "xml")


	articles=[]
	entries = soup.find_all("entry")
	for entry in entries:
	    title = entry.find("title").text
	    paper_id = entry.find("id").text
	    published = entry.find("published").text
	    updated = entry.find("updated").text
	    summary = entry.find("summary").text
	    author = [author.text for author in entry.find_all("author")]
	    comments = entry.find("arxiv:comment").text if entry.find("arxiv:comment") else ""
	    journal_ref = entry.find("arxiv:journal_ref").text if entry.find("arxiv:journal_ref") else ""
	    link = entry.find("link")["href"] if entry.find("link") else ""
	    primary_category = entry.find("arxiv:primary_category")["term"] if entry.find("arxiv:primary_category") else ""
	    categories = [cat["term"] for cat in entry.find_all("category")]
	    doi = entry.find("arxiv:doi").text if entry.find("arxiv:doi") else ""
	    license = entry.find("arxiv:license")["type"] if entry.find("arxiv:license") else ""
	    affiliation = [aff.text for aff in entry.find_all("arxiv:affiliation")]


	    # Append article information to the list
	    articles.append({
	        "Title": title,
	        "ID": paper_id,
	        "Published": published,
	        "Updated": updated,
	        "Summary": summary,
	        "Author": author,
	        "Comments": comments,
	        "Journal_Ref": journal_ref,
	        "Link": link,
	        "Primary_Category": primary_category,
	        "Categories": categories,
	        "DOI": doi,
	        "License": license,
	        "Affiliation": affiliation,
	    })


	df = pd.DataFrame(articles)

	return df





