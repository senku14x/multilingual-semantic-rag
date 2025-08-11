!pip -q install sentence-transformers faiss-cpu gradio datasets transformers accelerate wikipedia tqdm pandas

import os, json, numpy as np, faiss, torch, wikipedia, pandas as pd
from datetime import datetime
from typing import List, Dict
from collections import Counter
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import gradio as gr

PROJECT = "multilingual_semantic_rag"
ASSETS  = os.path.join(PROJECT, "assets")
os.makedirs(ASSETS, exist_ok=True)
LANGS  = ["en", "hi", "fr", "es"]
TOPICS = ["Biodiversity","Climate change","Renewable energy","Sustainable agriculture","Water scarcity","Public health","Artificial intelligence","Soil erosion","Food security","Waste management"]

def get_summary_for_topic(lang: str, topic: str, sentences: int = 3):
    wikipedia.set_lang(lang)
    try:
        text = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
        return text, topic
    except Exception:
        try:
            results = wikipedia.search(topic)
            for title in results[:3]:
                try:
                    text = wikipedia.summary(title, sentences=sentences, auto_suggest=False, redirect=True)
                    return text, title
                except Exception:
                    continue
        except Exception:
            pass
    return None, None

docs, doc_id = [], 1
for lang in LANGS:
    for topic in TOPICS:
        text, title = get_summary_for_topic(lang, topic, sentences=3)
        if not text:
            continue
        text = " ".join(text.split())
        if len(text) < 120:
            continue
        docs.append({"id": doc_id, "lang": lang, "title": title, "text": text})
        doc_id += 1

seen, unique_docs = set(), []
for d in docs:
    key = (d["lang"], d["text"])
    if key in seen:
        continue
    seen.add(key)
    unique_docs.append(d)
docs = unique_docs

dataset = Dataset.from_list(docs)
DATA_DIR = os.path.join(ASSETS, "wiki_dataset")
os.makedirs(DATA_DIR, exist_ok=True)
dataset.save_to_disk(DATA_DIR)
jsonl_path = os.path.join(ASSETS, "wiki_docs.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in docs:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

dataset = load_from_disk(DATA_DIR)
doc_texts  = dataset["text"]
doc_ids    = dataset["id"]
doc_langs  = dataset["lang"]
doc_titles = dataset["title"]

EMB_MODEL = "intfloat/multilingual-e5-base"
embedder  = SentenceTransformer(EMB_MODEL)

def embed_texts(texts: List[str], batch_size=64, is_query=False) -> np.ndarray:
    prefix = "query: " if is_query else "passage: "
    inputs = [prefix + t for t in texts]
    vecs = embedder.encode(inputs, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return vecs

doc_vecs = embed_texts(doc_texts, batch_size=64, is_query=False)
dim = doc_vecs.shape[1]
EMB_PATH  = os.path.join(ASSETS, "doc_vecs.npy")
META_PATH = os.path.join(ASSETS, "metadata.jsonl")
np.save(EMB_PATH, doc_vecs)
with open(META_PATH, "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        row = {"id": int(doc_ids[i]), "lang": doc_langs[i], "title": doc_titles[i], "text": doc_texts[i]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

index = faiss.IndexFlatIP(dim)
index.add(doc_vecs)
INDEX_PATH = os.path.join(ASSETS, "faiss.index")
faiss.write_index(index, INDEX_PATH)

def semantic_search(query: str, k: int = 5) -> List[Dict]:
    q_vec = embed_texts([query], is_query=True)
    D, I  = index.search(q_vec, k)
    scores = D[0].tolist()
    idxs   = [int(x) for x in I[0].tolist()]
    hits = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        hits.append({"rank": rank, "score": float(score), "doc_id": int(doc_ids[idx]), "lang": doc_langs[idx], "title": doc_titles[idx], "text": doc_texts[idx]})
    return hits

TRANS_NAME = "facebook/nllb-200-distilled-600M"
trans_tok = AutoTokenizer.from_pretrained(TRANS_NAME, src_lang="eng_Latn", tgt_lang="eng_Latn")
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_NAME).to("cuda" if torch.cuda.is_available() else "cpu").eval()
NLLB_CODE = {"en":"eng_Latn","fr":"fra_Latn","es":"spa_Latn","hi":"hin_Deva"}
_translation_cache = {}

def translate_once(text: str, src: str, tgt: str, max_new_tokens: int = 256) -> str:
    if src == tgt:
        return text
    key = (text, src, tgt)
    if key in _translation_cache:
        return _translation_cache[key]
    enc = trans_tok(text, return_tensors="pt", truncation=True).to(trans_model.device)
    out = trans_model.generate(**enc, forced_bos_token_id=trans_tok.convert_tokens_to_ids(NLLB_CODE[tgt]), max_new_tokens=max_new_tokens, num_beams=2, no_repeat_ngram_size=3)
    translated = trans_tok.batch_decode(out, skip_special_tokens=True)[0]
    _translation_cache[key] = translated
    return translated

GEN_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
gen_tok   = AutoTokenizer.from_pretrained(GEN_NAME)
dtype     = torch.float16 if torch.cuda.is_available() else torch.float32
gen_model = AutoModelForCausalLM.from_pretrained(GEN_NAME, torch_dtype=dtype).to("cuda" if torch.cuda.is_available() else "cpu").eval()
MAX_TOKENS_CONTEXT = 800
MAX_TOKENS_SNIPPET = 120
OUTPUT_BUDGET      = 220

def trim_snippet(snippet, max_tokens=MAX_TOKENS_SNIPPET):
    toks = gen_tok.encode(snippet, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return snippet
    return gen_tok.decode(toks[:max_tokens], skip_special_tokens=True)

def build_prompt(docs_translated, query, q_lang):
    bullets = "\n".join(f"- {t.strip()}" for t in docs_translated)
    return f"You are a precise multilingual assistant. Use ONLY the context to answer in {q_lang}.\nIf the context is insufficient, say you don't know.\n\nContext:\n{bullets}\n\nQuestion ({q_lang}): {query}\n\nAnswer ({q_lang}):"

@torch.inference_mode()
def generate_answer(prompt, max_new_tokens=OUTPUT_BUDGET):
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(gen_model.device)
    output_ids = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=gen_tok.eos_token_id, pad_token_id=gen_tok.eos_token_id)
    text = gen_tok.decode(output_ids[0], skip_special_tokens=True)
    return text[text.rfind("Answer"):].split(":", 1)[-1].strip()

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

def _short(text, n=160):
    return (text[:n] + "…") if len(text) > n else text

def app_pipeline(query, answer_lang, k):
    answer, hits = rag_answer(query, q_lang=answer_lang, k=int(k))
    lines = []
    for h in hits:
        lines.append(f"[{h['lang']}] {h['title']}  —  score={h['score']:.3f}\n  {_short(h['text'])}")
    sources = "\n\n".join(lines) if lines else "No sources found."
    return answer, sources

with gr.Blocks(title="Multilingual Semantic Search + RAG") as demo:
    gr.Markdown("## Multilingual Semantic Search + RAG")
    with gr.Row():
        query      = gr.Textbox(label="Your question")
        answer_lang= gr.Dropdown(choices=["en","hi","fr","es"], value="en", label="Answer language")
        k          = gr.Slider(1, 8, value=4, step=1, label="Top-k documents")
    run_btn  = gr.Button("Search & Answer")
    answer_o = gr.Textbox(label="Answer", lines=6)
    sources_o= gr.Textbox(label="Sources", lines=12)
    run_btn.click(app_pipeline, inputs=[query, answer_lang, k], outputs=[answer_o, sources_o])

demo.launch(share=True)
