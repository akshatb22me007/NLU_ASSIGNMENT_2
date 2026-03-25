import google.generativeai as genai
import time

# 🔑 Add your API key here
genai.configure(api_key="AIzaSyBOFd5Le9N7fWrSUfLI5_iAqSfqEo-VAUc")

model = genai.GenerativeModel("gemini-3-flash-preview")

# Prompt template
PROMPT = """
Generate a list of Indian names.

Rules:
- Only names
- One name per line
- No numbering
- No explanations
- Mix of male, female, and unisex names
- Avoid duplicates

Generate around 200 names.
"""

def generate_batch():
    response = model.generate_content(PROMPT)
    
    text = response.text
    
    names = []
    for line in text.split("\n"):
        name = line.strip().lower()
        
        # Filter valid names
        if name.isalpha() and len(name) > 2:
            names.append(name)
    
    return names


# Collect names
all_names = set()

while len(all_names) < 1000:
    try:
        new_names = generate_batch()
        all_names.update(new_names)

        print(f"Collected: {len(all_names)} names")

        time.sleep(2)  # avoid rate limit

    except Exception as e:
        print("Error:", e)
        time.sleep(5)


# Save file
with open("TrainingNames.txt", "w") as f:
    for name in sorted(all_names):
        f.write(name + "\n")

print("✅ Done! 1000 names saved.")