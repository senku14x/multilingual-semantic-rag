🌍 Multilingual Semantic Search + RAG
An end-to-end multilingual semantic search engine that lets users ask a question in any supported language (English, Hindi, French, Spanish) and retrieves cross-lingual relevant documents.
It then uses Retrieval-Augmented Generation (RAG) to generate a grounded, context-aware answer.

🚀 Features
Cross-Lingual Search — Find documents in multiple languages for a query in any one language.

Dense Embeddings — Uses intfloat/multilingual-e5-base for high-quality semantic representations.

Vector Search — Fast similarity search with FAISS.

RAG Integration — TinyLLaMA generates answers using retrieved documents.

Multilingual Translation Bridge — facebook/nllb-200 enables search and answer generation across languages.

Interactive UI — Built with Gradio; easily deployable to Hugging Face Spaces
