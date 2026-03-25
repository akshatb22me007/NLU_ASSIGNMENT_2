import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

# =========================
# CONFIG
# =========================

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
CORPUS_CANDIDATES = ["corpus.txt"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 7),
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 140,
    }
)

# Load models
cbow_model = Word2Vec.load("models/CBOW_dim100_win3_neg10.model")
sg_model = Word2Vec.load("models/SkipGram_dim100_win3_neg10.model")

# Words to visualize
words = [
    "student", "faculty", "research", "engineering",
    "btech", "mtech", "phd", "course", "exam", "algorithm", "computer",
    "project", "lab", "admission", "campus", "library", "sports", "culture",
]

# =========================
# HELPER FUNCTION
# =========================

def get_vectors(model, words):
    vectors = []
    labels = []
    missing = []

    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
            labels.append(w)
        else:
            missing.append(w)

    return np.array(vectors), labels, missing

# =========================
# PLOTTING FUNCTION
# =========================

def reduce_embeddings(vectors, method="tsne"):
    if method == "tsne":
        n_samples = len(vectors)
        if n_samples < 3:
            return vectors[:, :2]
        perplexity = max(2, min(8, n_samples - 1, n_samples // 2))
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
        )
    else:
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
        reducer = PCA(n_components=2, random_state=42)

    return reducer.fit_transform(vectors)


def set_axis_padding(ax, points, pad_ratio=0.12):
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    x_range = max(1e-6, x_vals.max() - x_vals.min())
    y_range = max(1e-6, y_vals.max() - y_vals.min())
    ax.set_xlim(x_vals.min() - x_range * pad_ratio, x_vals.max() + x_range * pad_ratio)
    ax.set_ylim(y_vals.min() - y_range * pad_ratio, y_vals.max() + y_range * pad_ratio)


def plot_on_axis(ax, reduced, labels, title, word_colors):
    offsets = [(7, 7), (7, -9), (-9, 7), (-9, -9), (10, 0), (-10, 0), (0, 10), (0, -10)]

    for i, word in enumerate(labels):
        x, y = reduced[i]
        color = word_colors[word]

        ax.scatter(
            x,
            y,
            s=80,
            c=[color],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
            zorder=2,
        )

        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            word,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=3,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    set_axis_padding(ax, reduced)


def plot_embeddings(model, words, method="tsne", title="plot", filename="plot.png", word_colors=None):
    vectors, labels, missing = get_vectors(model, words)

    if len(vectors) == 0:
        print(f"No valid words for {title}")
        return

    if missing:
        print(f"Missing words in {title}: {missing}")

    reduced = reduce_embeddings(vectors, method=method)

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_on_axis(ax, reduced, labels, title, word_colors)

    save_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Saved: {save_path}")


def build_word_colors(words):
    cmap = plt.cm.get_cmap("tab20", len(words))
    return {word: cmap(i) for i, word in enumerate(words)}


def plot_comparison_dashboard(cbow_model, sg_model, words, word_colors):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    cbow_vectors, cbow_labels, _ = get_vectors(cbow_model, words)
    sg_vectors, sg_labels, _ = get_vectors(sg_model, words)

    if len(cbow_vectors) > 0:
        plot_on_axis(
            axs[0, 0],
            reduce_embeddings(cbow_vectors, method="pca"),
            cbow_labels,
            "CBOW PCA",
            word_colors,
        )
        plot_on_axis(
            axs[0, 1],
            reduce_embeddings(cbow_vectors, method="tsne"),
            cbow_labels,
            "CBOW t-SNE",
            word_colors,
        )

    if len(sg_vectors) > 0:
        plot_on_axis(
            axs[1, 0],
            reduce_embeddings(sg_vectors, method="pca"),
            sg_labels,
            "Skip-gram PCA",
            word_colors,
        )
        plot_on_axis(
            axs[1, 1],
            reduce_embeddings(sg_vectors, method="tsne"),
            sg_labels,
            "Skip-gram t-SNE",
            word_colors,
        )

    fig.suptitle("Word2Vec Embedding Comparison", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    dashboard_path = os.path.join(PLOT_DIR, "embedding_dashboard.png")
    fig.savefig(dashboard_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {dashboard_path}")


def find_corpus_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def plot_word_cloud(corpus_path, filename="wordcloud_corpus.png"):
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        print(f"Corpus file is empty: {corpus_path}")
        return

    cloud = WordCloud(
        width=1800,
        height=1000,
        background_color="white",
        colormap="viridis",
        max_words=200,
        contour_width=0,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in Corpus", fontsize=18, fontweight="bold", pad=14)

    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")

# =========================
# GENERATE ALL PLOTS
# =========================

word_colors = build_word_colors(words)

# CBOW
plot_embeddings(
    cbow_model,
    words,
    method="tsne",
    title="CBOW t-SNE",
    filename="cbow_tsne.png",
    word_colors=word_colors,
)

plot_embeddings(
    cbow_model,
    words,
    method="pca",
    title="CBOW PCA",
    filename="cbow_pca.png",
    word_colors=word_colors,
)

# Skip-gram
plot_embeddings(
    sg_model,
    words,
    method="tsne",
    title="Skip-gram t-SNE",
    filename="skipgram_tsne.png",
    word_colors=word_colors,
)

plot_embeddings(
    sg_model,
    words,
    method="pca",
    title="Skip-gram PCA",
    filename="skipgram_pca.png",
    word_colors=word_colors,
)

plot_comparison_dashboard(cbow_model, sg_model, words, word_colors)

corpus_file = find_corpus_file(CORPUS_CANDIDATES)
if corpus_file:
    plot_word_cloud(corpus_file)
else:
    print("No corpus file found. Expected one of: corpus.txt, final_corpus.txt")

print("\n✅ All plots saved in 'plots/' folder")