from gensim.models import Word2Vec

model = Word2Vec.load("models/SkipGram_dim100_win5_neg10.model")

words = ["research", "student", "phd", "exam"]

for word in words:
    if word in model.wv:
        print(f"\nTop 5 words similar to '{word}':")
        
        similar_words = model.wv.most_similar(word, topn=5)
        
        for w, score in similar_words:
            print(f"{w} ({score:.4f})")
    else:
        print(f"\n'{word}' not in vocabulary")

def analogy(a, b, c):
    try:
        result = model.wv.most_similar(
            positive=[b, c],
            negative=[a],
            topn=3
        )
        
        print(f"\n{a} : {b} :: {c} : ?")
        for word, score in result:
            print(f"{word} ({score:.4f})")
            
    except KeyError as e:
        print(f"Word not in vocab: {e}")

# 1
analogy("ug", "btech", "pg")

# 2
analogy("student", "exam", "research")

# 3
analogy("btech", "undergraduate", "phd")

# 4 (extra, good one)
analogy("student", "class", "research")