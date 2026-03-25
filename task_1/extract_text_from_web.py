import requests
from bs4 import BeautifulSoup
import os

urls = [
    "https://iitj.ac.in",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/chemistry/en/chemistry"
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/electrical-engineering/",
    "https://www.iitj.ac.in/mathematics/",
    "https://www.iitj.ac.in/mechanical-engineering/",
    "https://www.iitj.ac.in/materials-engineering/en/materials-engineering",
    "https://www.iitj.ac.in/physics/",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://www.iitj.ac.in/Office-of-Academics/en/BS-Physics-with-Specialization",
    "https://www.iitj.ac.in/Office-of-Academics/en/BS-(Chemistry)-with-Specialization",
    "https://www.iitj.ac.in/Master-of-Technology/en/Master-of-Technology",
    "https://www.iitj.ac.in/chemistry/en/post-graduate-program",
    "https://www.iitj.ac.in/mathematics/en/m.sc.",
    "https://www.iitj.ac.in/physics/en/postgraduate-program",
    "https://www.iitj.ac.in/mathematics/en/doctoral-program",
    "https://www.iitj.ac.in/chemistry/en/doctoral-program",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/doctorial-program",
    "https://www.iitj.ac.in/computer-science-engineering/en/doctoral-programs",
    "https://www.iitj.ac.in/electrical-engineering/en/doctoral-program",
    "https://www.iitj.ac.in/mechanical-engineering/en/doctorial-program",
    "https://www.iitj.ac.in/chemical-engineering/en/doctoral-program",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/phd-programs",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/phd",
    "https://www.iitj.ac.in/school-of-liberal-arts/en/doctoral-program",
    "https://www.iitj.ac.in/itep/",
    "https://www.iitj.ac.in/office-of-executive-education/en/Program-Portfolio",
    "https://www.iitj.ac.in/school-of-design/en/masters-of-design-(mdes)-2025",
    "https://www.iitj.ac.in/es/en/engineering-science",
    "https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
    "https://www.iitj.ac.in/crf/en/crf",
    "https://anandmishra22.github.io/",
    "https://home.iitj.ac.in/~mvatsa/",
    "https://home.iitj.ac.in/~richa/",
    "https://3dcomputervision.github.io/",
    "https://sites.google.com/view/debasisdas/home",
    "https://sites.google.com/iitj.ac.in/dmishra",
    "https://sites.google.com/site/romibitsnbob/home",
    "https://home.iitj.ac.in/~vimalraj/",
    "https://home.iitj.ac.in/~palashdas/",    
]

output_folder = "data/web"
os.makedirs(output_folder, exist_ok=True)

for i, url in enumerate(urls):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract visible text
        text = soup.get_text(separator=" ")

        file_path = os.path.join(output_folder, f"doc{i+1}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved: {file_path}")

    except Exception as e:
        print(f"Error with {url}: {e}")