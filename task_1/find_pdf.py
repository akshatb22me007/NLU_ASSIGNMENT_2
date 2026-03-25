import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

visited = set()
saved_links = set()

BASE_DOMAIN = "iitj.ac.in"

# Open file once in append mode
output_file = open("pdf_links.txt", "a", encoding="utf-8")

def is_valid(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith(BASE_DOMAIN)

def find_pdfs(url, depth=2):
    if depth == 0 or url in visited:
        return

    if not is_valid(url):
        return

    visited.add(url)

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link['href']
            full_url = urljoin(url, href)

            # Skip unwanted links
            if not is_valid(full_url) or full_url.startswith("mailto:") or "#" in full_url:
                continue

            # If PDF → write immediately
            if full_url.lower().endswith(".pdf"):
                if full_url not in saved_links:
                    saved_links.add(full_url)

                    output_file.write(full_url + "\n")
                    output_file.flush()   # 🔥 real-time write

                    print(f"Saved: {full_url}")

            else:
                find_pdfs(full_url, depth - 1)

    except Exception as e:
        print(f"Error: {url}")

# Start
start_url = "https://iitj.ac.in"
find_pdfs(start_url, depth=2)

output_file.close()

print(f"\nTotal PDFs collected: {len(saved_links)}")