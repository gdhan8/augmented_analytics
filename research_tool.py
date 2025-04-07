import requests
from xml.etree import ElementTree
import os

def query_arxiv(search_query, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = search_query.replace(" ", "+")
    query = f"search_query={search_query}&start=0&max_results={max_results}"
    response = requests.get(base_url + query)

    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        results = []

        for entry in entries:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text
            results.append({
                "title": title.strip(),
                "summary": summary.strip(),
                "link": link.strip()
            })

        return results
    else:
        response.raise_for_status()

def download_papers(results, download_dir='papers'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for result in results:
        link = result['link']
        paper_id = link.split('/')[-1]
        pdf_url = f"http://arxiv.org/pdf/{paper_id}.pdf"
        response = requests.get(pdf_url)

        if response.status_code == 200:
            file_path = os.path.join(download_dir, f"{paper_id}.pdf")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {paper_id}.pdf, {result['title']}, {pdf_url}")
        else:
            print(f"Failed to download {paper_id}.pdf")

results = query_arxiv('', max_results=10)
download_papers(results, download_dir='papers')