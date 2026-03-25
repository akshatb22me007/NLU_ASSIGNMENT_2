# NLU Assignment 2

This repository contains two independent NLP tasks:

- `task_1`: Word2Vec pipeline on IITJ web + PDF text (data collection, preprocessing, training, evaluation, visualization)
- `task_2`: Character-level Indian name generation using RNN, BLSTM, and Attention RNN

## Quick Start

Use Python 3.9+ and create/activate your environment.

### Task 1 (Word2Vec)

```bash
cd task_1
pip install requests beautifulsoup4 pdfplumber nltk gensim pandas matplotlib scikit-learn wordcloud
python extract_text_from_web.py
python remove_iitj_header.py --input data/web --output data/web_clean
python find_pdf.py
python extract_text_from_pdf.py
python data_preprocess.py
python train.py
python eval.py
python visualization.py
```

### Task 2 (Name Generation)

```bash
cd task_2
pip install torch matplotlib google-generativeai
python generate_name.py
python main.py
```

## Outputs

- Task 1:
  - Trained Word2Vec models in `task_1/models/`
  - Evaluation summary in `task_1/word2vec_results.csv`
  - Visualizations in `task_1/plots/`
- Task 2:
  - Trained checkpoints in `task_2/models/`
  - Generated names and plots in `task_2/outputs/`

## Notes

- Detailed, step-by-step instructions are available in:
  - `task_1/README.md`
  - `task_2/README.md`
