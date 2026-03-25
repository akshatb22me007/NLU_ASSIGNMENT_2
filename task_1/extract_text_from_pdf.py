import pdfplumber
import requests
import os
from urllib.parse import urlparse

links_file = "data/pdf_links.txt"
output_folder = "data/web_clean"

os.makedirs(output_folder, exist_ok=True)

def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

with open(links_file, "r") as f:
    links = f.readlines()

for link in links:
    link = link.strip()

    try:
        print(f"Processing: {link}")

        # Download PDF
        response = requests.get(link, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download: {link}")
            continue

        # Temporary PDF file
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(response.content)

        # Extract text
        text = ""
        with pdfplumber.open(temp_pdf) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        # Create output filename
        filename = get_filename_from_url(link).replace(".pdf", ".txt")
        output_path = os.path.join(output_folder, filename)

        # Save text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error with {link}: {e}")