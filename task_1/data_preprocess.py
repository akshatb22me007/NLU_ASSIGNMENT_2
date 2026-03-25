import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

input_folder = "data/web_clean"
output_file = "corpus.txt"

stop_words = set(stopwords.words('english'))

# =========================
# STEP 1: REMOVE DUPLICATE DOCUMENTS
# =========================

seen_docs = set()
unique_texts = []

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            text = f.read().strip()

            if text not in seen_docs:
                seen_docs.add(text)
                unique_texts.append(text)

print(f"After doc deduplication: {len(unique_texts)} docs")

# =========================
# STEP 2: SENTENCE LEVEL DEDUP
# =========================

seen_sentences = set()
clean_sentences = []

for text in unique_texts:
    sentences = sent_tokenize(text)

    for sent in sentences:
        sent = sent.strip()

        if len(sent) < 20:   # remove very small junk
            continue

        if sent not in seen_sentences:
            seen_sentences.add(sent)
            clean_sentences.append(sent)

print(f"Unique sentences: {len(clean_sentences)}")

# =========================
# STEP 3: CLEAN + TOKENIZE
# =========================

all_tokens = []

for sent in clean_sentences:
    # Lowercase
    sent = sent.lower()

    # Remove non-English
    sent = re.sub(r'[^\x00-\x7F]+', ' ', sent)

    # Remove punctuation & numbers
    sent = re.sub(r'[^a-z\s]', ' ', sent)

    # Remove extra spaces
    sent = re.sub(r'\s+', ' ', sent)

    tokens = word_tokenize(sent)

    # Remove stopwords + short words
    tokens = [
        w for w in tokens
        if w not in stop_words and len(w) > 2
    ]

    if len(tokens):   # keep meaningful sentences only
        all_tokens.extend(tokens)

# =========================
# STEP 4: SAVE FINAL CORPUS
# =========================

with open(output_file, "w", encoding="utf-8") as f:
    f.write(" ".join(all_tokens))

# =========================
# STATS
# =========================

vocab = set(all_tokens)

print("\n✅ Deduplication + Preprocessing Complete")
print(f"Final Tokens: {len(all_tokens)}")
print(f"Vocabulary Size: {len(vocab)}")