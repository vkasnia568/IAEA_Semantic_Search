"""
IAEA Safeguards Document Analysis Tool
Prototype for SGIM Innovation Team
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# -------------------------------------------------
# Load models once and cache
# -------------------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return embedder, ner_model

embedder, ner_model = load_models()

# -------------------------------------------------
# Sample documents (safeguards context)
# -------------------------------------------------
DOCUMENTS = [
    {
        "id": 1,
        "title": "Iran Natanz Enrichment Update",
        "content": "Iran has informed the IAEA about new IR-6 centrifuges being installed at the Natanz enrichment facility. The advanced centrifuges can enrich uranium faster than previous models. IAEA inspectors are monitoring the situation."
    },
    {
        "id": 2,
        "title": "Brazil-Argentina Nuclear Cooperation",
        "content": "Brazil and Argentina have renewed their bilateral agreement for the peaceful use of nuclear energy. Both countries operate research reactors under full-scope IAEA safeguards. The agreement includes joint development of nuclear medicine facilities."
    },
    {
        "id": 3,
        "title": "IAEA Board Discusses Middle East Safeguards",
        "content": "The IAEA Board of Governors meeting discussed safeguards implementation in the Middle East. Member states expressed concerns about verification activities and called for greater transparency. The Director General will report back in three months."
    },
    {
        "id": 4,
        "title": "North Korea Yongbyon Reactor Activity",
        "content": "Satellite imagery shows new construction activity at North Korea's Yongbyon nuclear complex. The reactor appears to be operational based on thermal signatures. IAEA inspectors have not had access to the site since 2009."
    },
    {
        "id": 5,
        "title": "Russia-China Nuclear Energy Pact",
        "content": "Russia and China signed a strategic partnership for the construction of four new VVER-1200 reactors in China. The agreement includes technology transfer and joint research on fast neutron reactors. Both countries reaffirmed commitment to IAEA safeguards."
    }
]

# -------------------------------------------------
# Build vector index for similarity search
# -------------------------------------------------
@st.cache_resource
def build_index():
    texts = [doc["content"] for doc in DOCUMENTS]
    vectors = embedder.encode(texts)
    dim = vectors.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(vectors).astype('float32'))
    return idx

index = build_index()

# -------------------------------------------------
# Search function
# -------------------------------------------------
def search_documents(query, top_k=3):
    q_vec = embedder.encode([query])
    distances, indices = index.search(np.array(q_vec).astype('float32'), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc = DOCUMENTS[idx]
        score = float(1 / (1 + distances[0][i])) * 100
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": score
        })
    return results

# -------------------------------------------------
# Entity extraction
# -------------------------------------------------
def get_entities(text):
    raw_entities = ner_model(text)
    formatted = []
    for e in raw_entities:
        formatted.append({
            "entity": e['word'],
            "type": e['entity_group'],
            "conf": f"{e['score']:.1%}"
        })
    return formatted

# -------------------------------------------------
# Web Interface
# -------------------------------------------------
st.set_page_config(page_title="Document Analyzer", page_icon="⚛️", layout="wide")

st.title("⚛️ Safeguards Document Analysis")
st.caption("Semantic search and entity extraction prototype")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Document Set")
    st.caption(f"{len(DOCUMENTS)} documents loaded")
    for doc in DOCUMENTS:
        with st.expander(doc["title"]):
            st.write(doc["content"][:120] + "...")
    st.markdown("---")
    st.markdown("**Libraries used:**")
    st.markdown("- Sentence-Transformers")
    st.markdown("- FAISS")
    st.markdown("- BERT (NER)")
    st.markdown("- Streamlit")

# Main tabs
tab1, tab2 = st.tabs(["Search", "Entities"])

with tab1:
    st.subheader("Document Search")
    st.markdown("*Find documents by meaning, not just keywords*")
    
    query = st.text_input("Query:", placeholder="e.g., Middle East enrichment")
    
    if query:
        with st.spinner("Searching..."):
            results = search_documents(query)
        
        for r in results:
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{r['title']}**")
                    st.write(r['content'])
                with c2:
                    st.metric("Match", f"{r['score']:.1f}%")
                    st.progress(r['score'] / 100)
                st.markdown("---")

with tab2:
    st.subheader("Entity Extraction")
    st.markdown("*Identify organizations, locations, and names*")
    
    default_text = "Iran announced new centrifuges at Natanz while the IAEA Director General Rafael Grossi visited Tehran."
    text_input = st.text_area("Text:", value=default_text, height=150)
    
    if st.button("Extract"):
        with st.spinner("Processing..."):
            entities = get_entities(text_input)
        
        if entities:
            orgs = [e for e in entities if e['type'] == 'ORG']
            locs = [e for e in entities if e['type'] == 'LOC']
            pers = [e for e in entities if e['type'] == 'PER']
            
            if orgs:
                st.markdown("**Organizations**")
                for e in orgs:
                    st.write(f"- {e['entity']} ({e['conf']})")
            if locs:
                st.markdown("**Locations**")
                for e in locs:
                    st.write(f"- {e['entity']} ({e['conf']})")
            if pers:
                st.markdown("**Persons**")
                for e in pers:
                    st.write(f"- {e['entity']} ({e['conf']})")
        else:
            st.warning("No entities found.")

st.markdown("---")
st.caption("Prototype for SGIM Innovation Team internship application")