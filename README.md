# 🔍 AdaptCRAG — Corrective Retrieval Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.50-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Qdrant-Cloud-purple" />
  <img src="https://img.shields.io/badge/NVIDIA_NIM-API-76B900?logo=nvidia&logoColor=white" />
</p>

<p align="center">
  <b>A domain-configurable self-correcting RAG system that evaluates retrieval quality,
  routes intelligently across three confidence paths, and falls back to live web search —
  with full confidence transparency shown to the user.</b>
</p>

---

## 📌 What is CRAG?

CRAG implements the **Corrective RAG (CRAG)** architecture from Yan et al. (arXiv:2401.15884).
Unlike a standard RAG chatbot that blindly generates answers from retrieved text, AdaptCRAG
**evaluates retrieval quality first**, then takes different paths based on confidence:

- If the retrieved documents are **relevant** → answer directly
- If they are **partially relevant** → search the live web and combine
- If they are **irrelevant** → refuse to answer rather than hallucinate

The user always sees a confidence score and the exact path taken — complete transparency.

---

## 🎯 Key Differentiators

| Feature | Basic RAG | AdaptCRAG |
|---|---|---|
| Retrieval quality check | ❌ | ✅ LLM-based evaluator |
| Confidence scoring | ❌ | ✅ 0–100% with badge |
| Web search fallback | ❌ | ✅ Domain-filtered Tavily |
| Refuses bad answers | ❌ | ✅ LOW confidence → REFUSE |
| Multi-domain support | ❌ | ✅ 5 configurable domains |
| Source attribution | ❌ | ✅ Filename + page + URL |
| 100% free tier | ❌ | ✅ Zero infra cost |

---

## 🏗️ Architecture

```
User submits a query via Streamlit UI
               │
               ▼
   ┌─────────────────────────────────────────────┐
   │         CRAG PIPELINE (Python)              │
   │                                             │
   │  Step 1 ── Embed query                      │
   │             NVIDIA NIM (nv-embedqa-e5-v5)   │
   │             Query → 1024-dim vector         │
   │                    │                        │
   │  Step 2 ── Qdrant retrieval                 │
   │             Cosine similarity search        │
   │             Returns top-5 matching chunks   │
   │                    │                        │
   │  Step 3 ── Retrieval evaluator              │
   │             NVIDIA Gemma-4-31b scores chunks│
   │             Returns confidence 0.0 – 1.0    │
   │                    │                        │
   │  Step 4 ── Confidence router                │
   │             │                               │
   │       ┌─────┴──────────────────┐            │
   │       ▼            ▼           ▼            │
   │   HIGH (>70%)  MEDIUM (55-70%) LOW (<55%)   │
   │   GENERATE     WEB_SEARCH      REFUSE       │
   │   Use local    Tavily search   Don't answer │
   │   docs         + merge results              │
   │       └─────────────┬──────────┘            │
   │                     │                       │
   │  Step 5 ── Answer generator                 │
   │             NVIDIA Gemma-4-31b              │
   │             Grounded generation             │
   │                     │                       │
   └─────────────────────┼───────────────────────┘
                         │
                         ▼
         Streamlit UI — Answer + Confidence Badge + Sources
```

---

## 🔄 CRAG Decision Logic

```
Confidence > 0.70  →  🟢 HIGH    →  GENERATE
                       Use retrieved local documents directly.
                       Answer drawn entirely from your uploaded files.

Confidence 0.55–0.70  →  🟡 MEDIUM  →  WEB_SEARCH
                          Local docs insufficient.
                          Tavily searches domain-specific trusted sources.
                          Web results merged with local docs.
                          Answer generated from combined context.

Confidence < 0.55  →  🔴 LOW  →  REFUSE
                       Neither local docs nor web search are reliable.
                       System refuses to answer to prevent hallucination.
                       User advised to upload relevant docs or rephrase.
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | NVIDIA NIM — Gemma-4-31b-it | Answer generation + retrieval evaluation |
| **Embeddings** | NVIDIA NIM — nv-embedqa-e5-v5 | 1024-dim semantic vectors |
| **Vector DB** | Qdrant Cloud (free tier) | Vector storage and cosine similarity search |
| **Orchestration** | LangChain 0.3 | Document loading, text splitting, LLM chains |
| **Web Search** | Tavily API (free tier) | Domain-filtered live web fallback |
| **UI** | Streamlit | Interactive frontend with real-time pipeline trace |
| **Rate Limiter** | Custom `TavilyRateLimiter` | Persistent daily quota tracking (.json file) |
| **Config** | Pydantic Settings | Typed, validated environment variable management |
| **Logging** | Loguru | Rotating file + console structured logging |

**Total infrastructure cost: $0/month — entirely free tier.**

---

## 🌍 Local vs Cloud

```
Your Machine                    Cloud Services
────────────────────            ─────────────────────────────────
Python pipeline logic      ──→  NVIDIA NIM API   (embeddings + LLM)
Streamlit UI               ──→  Qdrant Cloud     (vector storage)
Tavily rate limiter        ──→  Tavily API        (web search)
LangChain orchestration
Loguru logging
```

No GPU required. No local model weights. Your machine only runs
Python and Streamlit — all heavy compute is handled by free cloud APIs.

---

## 📂 Project Structure

```
AdaptCRAG/
│
├── data/
│   └── raw/                        # Place your PDF / TXT documents here
│
├── src/
│   ├── core/
│   │   ├── data_ingestion.py       # PDF / TXT loading + chunking
│   │   ├── VectorEmbeddings.py     # NVIDIA NIM embedding generation
│   │   ├── evaluator.py            # LLM-based retrieval quality scorer
│   │   ├── router.py               # Confidence → GENERATE/WEB_SEARCH/REFUSE
│   │   ├── fallback_handler.py     # Web search decision + orchestration
│   │   ├── source_combiner.py      # Merge and rank local + web results
│   │   └── AnswerGenerator.py      # Final answer generation (3 templates)
│   │
│   ├── Integrations/
│   │   ├── Qdrant_client.py        # Qdrant Cloud CRUD + similarity search
│   │   └── tavily_search.py        # Domain-filtered Tavily wrapper
│   │
│   └── utils/
│       ├── config.py               # Pydantic settings — reads .env
│       ├── logger.py               # Loguru file + console handlers
│       └── rate_limiter.py         # Persistent Tavily daily quota counter
│
├── scripts/
│   └── prepare_data.py             # One-command ingestion pipeline
│
├── tests/
│   ├── test_retrieval.py           # Phase 2 — Qdrant retrieval tests
│   ├── test_evaluator.py           # Phase 3 — evaluator + router tests
│   ├── test_tavily_integration.py  # Phase 4 — web search tests
│   └── test_generator.py           # Phase 5 — answer generation tests
│
├── ui/
│   └── streamlit_app.py            # Streamlit frontend
│
├── logs/
│   └── CRAG.log                    # Auto-rotating log (500MB cap, 10-day retention)
│
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- Git

### 1. Clone

```bash
git clone https://github.com/Sansii18/AdaptCRAG.git
cd AdaptCRAG
```

### 2. Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get API Keys (All Free)

| Service | URL | Free Tier |
|---|---|---|
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) | 1000 credits/month |
| Qdrant Cloud | [cloud.qdrant.io](https://cloud.qdrant.io) | 1GB free cluster |
| Tavily | [app.tavily.com](https://app.tavily.com) | 1000 searches/month (~33/day) |

### 5. Configure Environment

```bash
cp .env.example .env
# Fill in your API keys
```

```bash
# .env
NVIDIA_API_KEY=""
NVIDIA_EMBEDDING_MODEL_ID=nvidia/nv-embedqa-e5-v5
NVIDIA_LLM_MODEL_ID=google/gemma-4-31b-it

QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=COLLECTION_NAME
QDRANT_VECTOR_SIZE=1024

TAVILY_API_KEY=""
TAVILY_SEARCH_DEPTH=advanced
TAVILY_MAX_RESULTS=5

RETRIEVAL_CONFIDENCE_THRESHOLD=0.70
WEB_SEARCH_THRESHOLD=0.55
TIMEOUT_SECONDS=120
```

### 6. Add Documents

```bash
cp your_documents/*.pdf data/raw/
```

### 7. Ingest Documents

```bash
python -m src.scripts.prepare_data
```

### 8. Launch

```bash
streamlit run ui/streamlit_app.py
# Opens at http://localhost:8501
```

---

## 🎮 How to Use

### Option A — Upload via UI (Recommended)

1. Open the app
2. Sidebar → **Upload Documents** → drag and drop PDF/TXT files
3. Click **⬆️ Ingest Documents**
4. Wait for success — sidebar shows updated chunk count
5. Select domain → type question → click **Get Answer**

### Option B — Bulk Ingest via CLI

```bash
python -m src.scripts.prepare_data --data_dir path/to/your/docs
```

### Select Domain

| Domain | Web Fallback Searches |
|---|---|
| `general` | Open web |
| `legal` | law.cornell.edu, justice.gov, courtlistener.com |
| `technical` | github.com, stackoverflow.com, docs.python.org |
| `financial` | sec.gov, reuters.com, ft.com, bloomberg.com |
| `academic` | arxiv.org, pubmed.ncbi.nlm.nih.gov, semanticscholar.org |

### Pipeline Trace

Every query shows a real-time trace:

```
✅ Step 1: Retrieved 5 documents
✅ Step 2: Confidence = 0.87
✅ Step 3: Action = GENERATE
✅ Step 4: Answer generated from local knowledge base
```

---

## 🧪 Running Tests

```bash
# Retrieval tests — no API call
pytest src/tests/test_retrieval.py -v

# Evaluator + router tests
pytest src/tests/test_evaluator.py -v -m "not integration"
pytest src/tests/test_evaluator.py -v -m "integration"       # uses NVIDIA API

# Web search tests
pytest src/tests/test_tavily_integration.py -v -m "not integration"
pytest src/tests/test_tavily_integration.py -v -m "integration"  # uses Tavily

# Answer generation tests
pytest src/tests/test_generator.py -v -m "not integration"
pytest src/tests/test_generator.py -v -m "integration"        # uses NVIDIA API

# All tests
pytest -v
```

---

## 📊 Performance & Limits

| Metric | Value |
|---|---|
| Embedding model MTEB score | 69.1 (nv-embedqa-e5-v5) |
| Vector dimensions | 1024 |
| Chunks retrieved per query | Top-5 by cosine similarity |
| Evaluation latency | ~3–8 seconds |
| Generation latency | ~5–30 seconds (Gemma-4-31b) |
| Max documents (free tier) | ~50,000 chunks (1GB Qdrant) |
| Tavily quota | 33 searches/day — tracked persistently |

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_LLM_MODEL_ID` | `google/gemma-4-31b-it` | LLM for generation and evaluation |
| `NVIDIA_EMBEDDING_MODEL_ID` | `nvidia/nv-embedqa-e5-v5` | Embedding model |
| `QDRANT_VECTOR_SIZE` | `1024` | Must match embedding model output dim |
| `QDRANT_COLLECTION_NAME` | `CRAG_PROJECT` | Qdrant collection name |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.70` | Above → GENERATE |
| `WEB_SEARCH_THRESHOLD` | `0.55` | Above (and below 0.70) → WEB_SEARCH |
| `TAVILY_SEARCH_DEPTH` | `advanced` | `basic` or `advanced` |
| `TAVILY_MAX_RESULTS` | `5` | Web results per search |
| `TIMEOUT_SECONDS` | `120` | LLM API call timeout |

---

## 📋 Key Dependencies

```
langchain==0.3.x                     # Orchestration
langchain-openai==0.3.x              # OpenAI-compatible LLM interface
langchain-nvidia-ai-endpoints        # NVIDIA NIM integration
qdrant-client                        # Qdrant vector database client
tavily-python                        # Tavily web search SDK
streamlit                            # UI framework
pydantic-settings                    # Typed config management
loguru                               # Structured logging
pypdf                                # PDF text extraction
python-dotenv                        # Environment variable loading
```

---

## 📚 Research Foundation

This project implements the CRAG architecture from:

> **Corrective Retrieval Augmented Generation**
> Yan et al., 2024 — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

Key implementation additions over the original paper:
- Domain-configurable trusted source filtering for web fallback
- Three-tier confidence badge display (not in original)
- Persistent Tavily rate limiting across sessions
- Runtime domain switching without pipeline restart
- Document upload directly via UI

---
<!-- 
## 🗺️ Roadmap

- [ ] LangGraph workflow graph (visual pipeline inspection)
- [ ] Answer verification / hallucination detection module
- [ ] Streamlit Cloud deployment
- [ ] Conversation history and memory
- [ ] Page-level citation with PDF highlights
- [ ] GraphRAG for complex multi-hop queries -->

---

## 👤 Author

**Sanskar Krishnani**

- 🧑🏼‍💻 GitHub: [@Sansii18](https://github.com/Sansii18)
- 🔗 LinkedIn: [linkedin.com/in/sanskar-krishnani](https://linkedin.com/in/sanskar-krishnani)

---

## Acknowledgements

- [Yan et al., 2024](https://arxiv.org/abs/2401.15884) — CRAG original paper
- [NVIDIA NIM](https://build.nvidia.com) — free cloud LLM and embedding APIs
- [Qdrant](https://qdrant.tech) — vector database
- [Tavily](https://tavily.com) — AI-optimised web search API
- [LangChain](https://langchain.com) — orchestration framework

---

<p align="center">
  Built by Sanskar Krishnani &nbsp;·&nbsp; Star this repo if it helped you! 🫶🏻
</p>