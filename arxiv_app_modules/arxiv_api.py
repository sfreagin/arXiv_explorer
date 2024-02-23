import requests
import pandas as pd
from bs4 import BeautifulSoup
import io 
from PyPDF2 import PdfReader  


def arxiv_query(chosen_category, start_date, end_date):

	max_results=1000
	start_date=start_date
	end_date=end_date
	chosen_category=chosen_category

	#https://groups.google.com/g/arxiv-api/c/mAFYT2VRpK0
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


def download_pdf_from_link(link):
    response = requests.get(link, stream=True)  # Making a GET request to download the PDF file

    if response.status_code == 200:  # Checking if the request was successful
        return io.BytesIO(response.content)  # Returning BytesIO object containing the PDF content

    else:
        print(f"Failed to download PDF from {link}")  # Printing error message if download fails
        return None


def extract_text_from_pdf(pdf_io):
    if pdf_io:
        try:
            reader = PdfReader(pdf_io)  # Creating a PdfReader object with the PDF content
            text = ""
        
            for page in reader.pages:  # Looping through each page in the PDF
                text += page.extract_text() + "\n"  # Extracting text from the page and appending it to the 'text' variable
            return text  # Returning the extracted text
        
        except Exception as e:
            print(f"Error occurred while extracting text from PDF: {str(e)}")  # Printing error message if extraction fails
            return ""




