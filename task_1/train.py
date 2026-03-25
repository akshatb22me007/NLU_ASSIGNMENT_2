import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import pandas as pd

nltk.download('punkt')

input_file = "corpus.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

sentences = []

for sent in sent_tokenize(text):
    tokens = word_tokenize(sent)
    
    if len(tokens) > 3:   # keep meaningful sentences
        sentences.append(tokens)

print("Total sentences:", len(sentences))
print("Sample:", sentences[0][:10])

# =========================
# PARAMETERS
# =========================

embedding_sizes = [50, 100, 200]
window_sizes = [3, 5, 8]
negative_samples = [5, 10]

results = []

# =========================
# TRAINING LOOP
# =========================

for model_type in ["CBOW", "SkipGram"]:
    sg = 0 if model_type == "CBOW" else 1

    for emb in embedding_sizes:
        for window in window_sizes:
            for neg in negative_samples:

                print(f"\nTraining {model_type} | dim={emb}, window={window}, neg={neg}")

                model = Word2Vec(
                    sentences=sentences,
                    vector_size=emb,
                    window=window,
                    negative=neg,
                    sg=sg,
                    min_count=3,
                    workers=4,
                    epochs=10
                )

                # =========================
                # EVALUATION
                # =========================

                test_word = None
                for w in ["student", "research", "engineering", "data"]:
                    if w in model.wv:
                        test_word = w
                        break

                if test_word:
                    similar = model.wv.most_similar(test_word, topn=5)
                else:
                    similar = []

                # Save results
                results.append({
                    "Model": model_type,
                    "Embedding": emb,
                    "Window": window,
                    "Negative": neg,
                    "Test Word": test_word,
                    "Top Words": str(similar)
                })

                # Save model
                model.save(f"models/{model_type}_dim{emb}_win{window}_neg{neg}.model")

# =========================
# SAVE RESULTS
# =========================

df = pd.DataFrame(results)
df.to_csv("word2vec_results.csv", index=False)

print("\n✅ Training Complete")