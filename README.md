# IAEA_Semantic_Search
Semantic document search with Streamlit and BERT NER
# IAEA Semantic Document Search

A smarter document search tool using sentence transformers and BERT-based entity extraction. Built with Streamlit for an interactive web interface.

## What it does

- **Semantic Search:** Finds documents by meaning, not just exact keywords.
- **Entity Extraction:** Identifies organizations, locations, and persons in text.
- **Web UI:** Clean interface with search and analysis tabs.

## Tech Stack

- Streamlit (frontend)
- Sentence-Transformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS for fast similarity search
- Hugging Face Transformers (`dslim/bert-base-NER`) for entity recognition

## How to Run

1. Clone this repo.
2. Install dependencies:
3. 3. Launch the app:
   4. 4. Open your browser at `http://localhost:8501`.

## Sample Documents (Hardcoded)

- Iran Natanz Enrichment Update
- Brazil-Argentina Nuclear Cooperation
- IAEA Board on Middle East Safeguards
- North Korea Yongbyon Reactor Activity
- Russia-China Nuclear Energy Pact

## Why Two Prototypes?

I also built a pure‑Python keyword search tool (see `IAEA_Keyword_Search`). That one runs anywhere without internet. This one shows what's possible with modern NLP libraries — better accuracy and a polished UI, but requires installing a few packages.

---

*Built for the SGIM internship application.*
