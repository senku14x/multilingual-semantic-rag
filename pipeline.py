# Install dependencies with FAISS-CPU (works reliably in Colab)
!pip install -q sentence-transformers faiss-cpu gradio datasets transformers accelerate

import os, torch, numpy as np, faiss
from datetime import datetime

# Check GPU availability
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

if gpu_available:
    print("GPU available but using FAISS-CPU (still very fast for semantic search)")
else:
    print("Using FAISS-CPU (optimal for most RAG applications)")

# Create project folders
PROJECT = "multilingual_semantic_rag"
ASSETS = os.path.join(PROJECT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# Languages in scope for MVP
LANGS = ["en", "hi", "fr", "es"]

print(" Setup complete", datetime.now().isoformat())
!python -V

from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/semantic_rag"  # Change path if you like
os.makedirs(BASE, exist_ok=True)
print("Saving project assets to:", BASE)

# Install the lightweight Wikipedia client (only once)
!pip -q install wikipedia

import os, json
from collections import Counter
import wikipedia                          # Python client for Wikipedia
from datasets import Dataset               # Nice tabular/columnar container
import pandas as pd

# --- Safety: fallbacks if Step 1 variables aren't present (lets this cell run standalone too) ---
if "LANGS" not in globals():              # languages we want to cover
    LANGS = ["en", "hi", "fr", "es"]
if "ASSETS" not in globals():             # where to save files
    ASSETS = "multilingual_semantic_rag/assets"
    os.makedirs(ASSETS, exist_ok=True)

# --- 2.1 Topics to fetch across languages ---
# Keep topics high-level so most Wikipedias have a page/redirect.
TOPICS = [
    "Biodiversity",
    "Climate change",
    "Renewable energy",
    "Sustainable agriculture",
    "Water scarcity",
    "Public health",
    "Artificial intelligence",
    "Soil erosion",
    "Food security",
    "Waste management",
]

# --- 2.2 Helper: fetch a short summary for (language, topic) robustly ---
def get_summary_for_topic(lang: str, topic: str, sentences: int = 3):
    """
    Tries to retrieve a short summary paragraph for `topic` in the given `lang`.
    1) Attempt the exact title (with autosuggest/redirects).
    2) If that fails (title differs across languages), search and try the top few results.
    Returns: (summary_text, resolved_title) or (None, None) if nothing found.
    """
    wikipedia.set_lang(lang)  # switch the client to that language’s wiki
    try:
        text = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
        return text, topic
    except Exception:
        # Title likely differs in this language — try search results
        try:
            results = wikipedia.search(topic)            # list of possible titles in that language
            for title in results[:3]:                    # try a few to stay fast
                try:
                    text = wikipedia.summary(title, sentences=sentences, auto_suggest=False, redirect=True)
                    return text, title
                except Exception:
                    continue
        except Exception:
            pass
    return None, None

# --- 2.3 Build the multilingual document list ---
docs = []
doc_id = 1

for lang in LANGS:
    for topic in TOPICS:
        text, resolved_title = get_summary_for_topic(lang, topic, sentences=3)
        if not text:
            # Skip silently if a topic isn't available in that language
            continue
        # Minimal cleaning: strip whitespace and keep only non-trivial passages
        text = " ".join(text.split())
        if len(text) < 120:  # skip very short stubs
            continue
        docs.append({
            "id": doc_id,
            "lang": lang,
            "title": resolved_title,  # the title we actually used in that language
            "text": text
        })
        doc_id += 1

print(f"Collected {len(docs)} passages across {len(LANGS)} languages and {len(TOPICS)} topics.")

# --- 2.4  Deduplicate exact duplicate texts within language ---
seen = set()
unique_docs = []
for d in docs:
    key = (d["lang"], d["text"])
    if key in seen:
        continue
    seen.add(key)
    unique_docs.append(d)

docs = unique_docs
print(f"After de-duplication: {len(docs)} passages.")

# --- 2.5 Wrap into a Hugging Face Dataset (easier downstream) ---
dataset = Dataset.from_list(docs)

# --- 2.6 Quick diagnostics ---
counts_by_lang = Counter(dataset["lang"])
print("Counts per language:", dict(counts_by_lang))

# Peek a few rows (pretty table)
display(pd.DataFrame(docs).head(8))

# --- 2.7 Persist to disk so later cells (embeddings/index) can reload quickly ---
DATA_DIR = os.path.join(ASSETS, "wiki_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

# Save as HF Dataset (Arrow format)
dataset.save_to_disk(DATA_DIR)

# Also save as JSONL for transparency/debugging
jsonl_path = os.path.join(ASSETS, "wiki_docs.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in docs:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved HF dataset to:", DATA_DIR)
print("Saved JSONL to:", jsonl_path)

import os, numpy as np, faiss, math, json
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ---------- 3.1 Load the dataset we saved in Step 2 ----------
# If you ran Step 2, DATA_DIR and ASSETS should already exist.
if "ASSETS" not in globals():
    ASSETS = "multilingual_semantic_rag/assets"
DATA_DIR = os.path.join(ASSETS, "wiki_dataset")
assert os.path.isdir(DATA_DIR), "Dataset directory not found. Run Step 2 first."

dataset = load_from_disk(DATA_DIR)
print("Loaded rows:", len(dataset), "| columns:", dataset.column_names)

# ---------- 3.2 Pick a multilingual embedding model ----------
# Reliable on Colab and strong for cross-ling retrieval.
MODEL_NAME = "intfloat/multilingual-e5-base"
embedder = SentenceTransformer(MODEL_NAME)  # sentence-transformers handles device automatically

# ---------- 3.3 Helper: batch encode with E5's recommended prefixes ----------
def embed_texts(texts, batch_size=64, is_query=False):
    """
    Encodes a list of strings into L2-normalized vectors.
    E5 expects 'query: ...' for queries and 'passage: ...' for documents.
    """
    prefix = "query: " if is_query else "passage: "
    # Prepend prefix once per text to align with model's training prompt
    inputs = [prefix + t for t in texts]

    # encode(..., normalize_embeddings=True) ensures vectors are L2-normalized
    vecs = embedder.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return vecs  # shape: (N, dim), dtype=float32 by default

# ---------- 3.4 Compute document embeddings ----------
doc_texts = dataset["text"]
doc_ids   = dataset["id"]       # keep IDs, we'll need them after FAISS search
doc_langs = dataset["lang"]     # handy metadata
doc_titles= dataset["title"]

doc_vecs = embed_texts(doc_texts, batch_size=64, is_query=False)
dim = doc_vecs.shape[1]
print(f"Embedding dim = {dim}, num docs = {len(doc_vecs)}")

# ---------- 3.5 Save embeddings + metadata to disk (for reuse) ----------
EMB_PATH = os.path.join(ASSETS, "doc_vecs.npy")
np.save(EMB_PATH, doc_vecs)

META_PATH = os.path.join(ASSETS, "metadata.jsonl")
with open(META_PATH, "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        row = {
            "id": int(doc_ids[i]),
            "lang": doc_langs[i],
            "title": doc_titles[i],
            "text": doc_texts[i],
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved embeddings ->", EMB_PATH)
print("Saved metadata   ->", META_PATH)

# ---------- 3.6 Build a FAISS index ----------
# We use IndexFlatIP (inner product). Because our vectors are normalized,
# IP is equivalent to cosine similarity.
index = faiss.IndexFlatIP(dim)
index.add(doc_vecs)  # add all document vectors to the index
print("FAISS ntotal =", index.ntotal)

# Persist the index to disk so you don't have to rebuild every session
INDEX_PATH = os.path.join(ASSETS, "faiss.index")
faiss.write_index(index, INDEX_PATH)
print("Wrote FAISS index ->", INDEX_PATH)

def search_raw(query_text, k=5):
    # encode query with E5's query prefix
    q_vec = embed_texts([query_text], is_query=True)

    # FAISS returns NumPy arrays (scores D, indices I)
    D, I = index.search(q_vec, k)

    # convert to plain Python types
    scores = D[0].tolist()                 # list of floats
    idxs   = [int(x) for x in I[0].tolist()]  # list of Python ints

    hits = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        hits.append({
            "rank":  rank,
            "score": float(score),
            "id":    int(doc_ids[idx]),
            "lang":  doc_langs[idx],
            "title": doc_titles[idx],
            "text":  (doc_texts[idx][:200] + ("..." if len(doc_texts[idx]) > 200 else ""))
        })
    return hits

# try again
for q in ["biodiversity importance", "खाद्य सुरक्षा क्या है? (What is food security?)"]:
    print("\nQuery:", q)
    for h in search_raw(q, k=3):
        print(f"  [{h['rank']}] score={h['score']:.3f} | {h['lang']} | {h['title']}")
        print("      ", h["text"])


from typing import List, Dict

# Keep these globals around from earlier steps:
# - embed_texts (adds E5 prefixes + normalizes)
# - index (FAISS)
# - doc_ids, doc_langs, doc_titles, doc_texts (metadata arrays)

def semantic_search(query: str, k: int = 5) -> List[Dict]:
    """
    Run cross-lingual semantic search.
    Returns a list of dicts with score + metadata, sorted by score desc.
    """
    # 1) Encode the query with E5's 'query:' prefix and L2-normalization
    q_vec = embed_texts([query], is_query=True)      # shape: (1, dim)

    # 2) FAISS nearest neighbors (exact search on normalized vectors → cosine)
    D, I = index.search(q_vec, k)                    # D: similarity scores, I: indices into doc arrays

    # 3) Convert to Python types & assemble a clean record per hit
    scores = D[0].tolist()
    idxs   = [int(x) for x in I[0].tolist()]

    hits = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        hits.append({
            "rank":  rank,
            "score": float(score),
            "doc_id": int(doc_ids[idx]),
            "lang":   doc_langs[idx],
            "title":  doc_titles[idx],
            "text":   doc_texts[idx],   # full text (RAG will chunk/translate as needed)
        })

    return hits

# ---- quick smoke test ----
for q in ["biodiversity importance", "खाद्य सुरक्षा क्या है?"]:
    print("\nQuery:", q)
    for h in semantic_search(q, k=3):
        print(f"[{h['rank']}] {h['score']:.3f} | {h['lang']} | {h['title']}")

# Install transformers once (already installed earlier), then load NLLB-200 distilled
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

# 5.1 — Model & tokenizer
TRANS_NAME = "facebook/nllb-200-distilled-600M"
trans_tok = AutoTokenizer.from_pretrained(TRANS_NAME, src_lang="eng_Latn", tgt_lang="eng_Latn")
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_NAME).to("cuda" if torch.cuda.is_available() else "cpu").eval()

# 5.2 — Manual mapping for the languages we care about
NLLB_CODE = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "hi": "hin_Deva",
}

# get ID for target BOS token
def get_lang_id(lang_code: str) -> int:
    return trans_tok.convert_tokens_to_ids(lang_code)

_translation_cache = {}

def translate_once(text: str, src: str, tgt: str, max_new_tokens: int = 256) -> str:
    if src == tgt:
        return text
    key = (text, src, tgt)
    if key in _translation_cache:
        return _translation_cache[key]

    enc = trans_tok(text, return_tensors="pt", truncation=True).to(trans_model.device)
    out = trans_model.generate(
        **enc,
        forced_bos_token_id=get_lang_id(NLLB_CODE[tgt]),
        max_new_tokens=max_new_tokens,
        num_beams=2,
        no_repeat_ngram_size=3
    )
    translated = trans_tok.batch_decode(out, skip_special_tokens=True)[0]
    _translation_cache[key] = translated
    return translated




def translate_batch(texts: List[str], srcs: List[str], tgt: str) -> List[str]:
    """
    Batched translation where each text may have its own source language.
    We loop (since sources differ), but you still benefit from model warm state.
    """
    out = []
    for t, s in zip(texts, srcs):
        out.append(translate_once(t, s, tgt))
    return out

# 5.4 — Quick smoke test: take top hits and translate to Hindi
hits = semantic_search("renewable energy benefits", k=3)
texts = [h["text"] for h in hits]
srcs  = [h["lang"] for h in hits]
to_hi = translate_batch(texts, srcs, tgt="hi")

for i, (h, t) in enumerate(zip(hits, to_hi), 1):
    print(f"\nHit {i}: {h['lang']} → hi | {h['title']}")
    print("Translated snippet:", t[:200] + ("..." if len(t) > 200 else ""))
# =========================================
# STEP 6 — RAG generator with token budget
# =========================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 6.1 — Load TinyLLaMA Chat model
GEN_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
gen_tok   = AutoTokenizer.from_pretrained(GEN_NAME)
dtype     = torch.float16 if torch.cuda.is_available() else torch.float32
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_NAME, torch_dtype=dtype
).to("cuda" if torch.cuda.is_available() else "cpu").eval()

# Safe budgets for Colab
MAX_TOKENS_CONTEXT = 800   # for all context snippets combined
MAX_TOKENS_SNIPPET = 120   # per snippet cap
OUTPUT_BUDGET      = 220   # answer tokens

# 6.2 Snippet trimming
def trim_snippet(snippet, max_tokens=MAX_TOKENS_SNIPPET):
    toks = gen_tok.encode(snippet, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return snippet
    return gen_tok.decode(toks[:max_tokens], skip_special_tokens=True)

# 6.3 Prompt builder
def build_prompt(docs_translated, query, q_lang):
    bullets = "\n".join(f"- {t.strip()}" for t in docs_translated)
    return (
f"You are a precise multilingual assistant. Use ONLY the context to answer in {q_lang}.\n"
f"If the context is insufficient, say you don't know.\n\n"
f"Context:\n{bullets}\n\n"
f"Question ({q_lang}): {query}\n\n"
f"Answer ({q_lang}):"
    )

# 6.4 Generator
@torch.inference_mode()
def generate_answer(prompt, max_new_tokens=OUTPUT_BUDGET):
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(gen_model.device)
    output_ids = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=gen_tok.eos_token_id,
        pad_token_id=gen_tok.eos_token_id
    )
    text = gen_tok.decode(output_ids[0], skip_special_tokens=True)
    return text[text.rfind("Answer"):].split(":", 1)[-1].strip()

# 6.5 Full RAG pipeline
def rag_answer(query: str, q_lang: str = "en", k: int = 4):
    hits = semantic_search(query, k=k)

    processed, total_tokens = [], 0
    for h in hits:
        txt = translate_once(h["text"], h["lang"], q_lang) if h["lang"] != q_lang else h["text"]
        txt = trim_snippet(txt)
        tok_count = len(gen_tok.encode(txt, add_special_tokens=False))
        if total_tokens + tok_count <= MAX_TOKENS_CONTEXT:
            processed.append(txt)
            total_tokens += tok_count
        else:
            break

    prompt = build_prompt(processed, query, q_lang)
    answer = generate_answer(prompt)
    return answer, hits

# 6.6 test
ans, hits = rag_answer("नवीकरणीय ऊर्जा के क्या लाभ हैं?", q_lang="hi", k=4)
print("RAG Answer:\n", ans, "\n")
print("Sources:")
for h in hits:
    print(f"- [{h['lang']}] {h['title']}")


import gradio as gr

# small helper to show compact source snippets
def _short(text, n=160):
    return (text[:n] + "…") if len(text) > n else text

def app_pipeline(query, answer_lang, k):
    # Runing the  full RAG pipeline
    answer, hits = rag_answer(query, q_lang=answer_lang, k=int(k))

    # Building a readable sources panel
    lines = []
    for h in hits:
        lines.append(f"[{h['lang']}] {h['title']}  —  score={h['score']:.3f}\n  {_short(h['text'])}")
    sources = "\n\n".join(lines) if lines else "No sources found."

    return answer, sources

with gr.Blocks(title="Multilingual Semantic Search + RAG") as demo:
    gr.Markdown("## 🌍 Multilingual Semantic Search + RAG\nType a question in any language. The app retrieves multilingual docs and answers in your chosen language.")
    with gr.Row():
        query      = gr.Textbox(label="Your question", placeholder="e.g., ¿Qué es la biodiversidad?")
        answer_lang= gr.Dropdown(choices=["en","hi","fr","es"], value="en", label="Answer language")
        k          = gr.Slider(1, 8, value=4, step=1, label="Top-k documents")

    run_btn  = gr.Button("Search & Answer 🚀")
    answer_o = gr.Textbox(label="Answer", lines=6)
    sources_o= gr.Textbox(label="Sources (original language)", lines=12)

    run_btn.click(app_pipeline, inputs=[query, answer_lang, k], outputs=[answer_o, sources_o])

demo.launch(share=True)

