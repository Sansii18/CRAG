# import streamlit as st
# import sys
# sys.path.insert(0, '.')

# from src.core.VectorEmbeddings import GenerateEmbeddings      # ← correct class name
# from src.Integrations.Qdrant_client import StoreQdrant         # ← correct class name
# from src.core.evaluator import RetrievalEvaluator
# from src.core.router import ConditionalRouer
# from src.core.fallback_handler import FallbackHandler
# from src.core.AnswerGenerator import AnswerGenerator           # ← correct module name
# from src.utils.tavily_rate_limiter import TavilyRateLimiter
# from src.utils.logger import logger
# from src.core.data_ingestion import DataIngestion

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config( # type: ignore
#     page_title="CRAG — Corrective RAG",
#     page_icon="🔍",
#     layout="wide"
# )

# st.title("🔍 CRAG — Corrective Retrieval Augmented Generation") # type: ignore
# st.markdown("*Intelligent question answering with confidence scoring and web search fallback*") # type: ignore

# # ── Cache heavy components that don't depend on domain ────────────────────────
# @st.cache_resource # type: ignore
# def init_base_components():
#     """Initialise components that are domain-independent — cached once"""
#     embedder = GenerateEmbeddings()
#     qdrant   = StoreQdrant()
#     router   = ConditionalRouer({})
#     fallback = FallbackHandler()
#     return embedder, qdrant, router, fallback

# # ── Document Upload ───────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("---")
#     st.subheader("📄 Upload Documents")

#     uploaded_files = st.file_uploader(
#         "Upload PDF or TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         help="Upload documents to add to the knowledge base"
#     )

#     if uploaded_files:
#         if st.button("⬆️ Ingest Documents", use_container_width=True):
#             import tempfile
#             import os
#             from src.core.data_ingestion import DocumentIngestor # type: ignore

#             ingestor = DocumentIngestor()
#             all_chunks = []

#             progress = st.progress(0)
#             status   = st.empty()

#             for i, file in enumerate(uploaded_files):
#                 # Write uploaded file to a temp path so loaders can read it
#                 suffix = ".pdf" if file.type == "application/pdf" else ".txt"
#                 with tempfile.NamedTemporaryFile(
#                     delete=False, suffix=suffix
#                 ) as tmp:
#                     tmp.write(file.read())
#                     tmp_path = tmp.name

#                 status.text(f"Processing: {file.name}")
#                 try:
#                     docs   = ingestor.load_data(tmp_path)       # ← your method name
#                     chunks = ingestor.split_data(docs)          # ← your method name
#                     all_chunks.extend(chunks)
#                 except Exception as e:
#                     st.error(f"Failed to process {file.name}: {e}")
#                 finally:
#                     os.unlink(tmp_path)                         # clean up temp file

#                 progress.progress((i + 1) / len(uploaded_files))

#             if all_chunks:
#                 status.text("Generating embeddings and uploading to Qdrant...")
#                 try:
#                     chunk_texts = [c.page_content for c in all_chunks]
#                     embeddings  = embedder.embed_chunks(chunk_texts) # type: ignore
#                     qdrant.create_collection(force_recreate=False) # type: ignore
#                     qdrant.upload_embeddings(chunk_texts, embeddings) # type: ignore

#                     st.success(
#                         f"✅ {len(all_chunks)} chunks from "
#                         f"{len(uploaded_files)} file(s) added to knowledge base"
#                     )
#                     # Clear cache so sidebar status updates
#                     st.cache_resource.clear()
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Upload failed: {e}")
                    
# # ── Cache evaluator and generator per domain ─────────────────────────────────
# @st.cache_resource # type: ignore
# def get_evaluator(domain: str) -> RetrievalEvaluator:
#     """Cached per domain — switching domain creates a new cached instance"""
#     return RetrievalEvaluator(domain=domain)

# @st.cache_resource # type: ignore
# def get_generator(domain: str) -> AnswerGenerator:
#     """Cached per domain"""
#     return AnswerGenerator(domain=domain)

# embedder, qdrant, router, fallback = init_base_components()

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar: # type: ignore
#     st.header("⚙️ Configuration") # type: ignore

#     # ✅ medical removed, general + others added
#     domain = st.selectbox( # type: ignore
#         "Select Domain",
#         ["general", "legal", "technical", "financial", "academic"],
#         index=0
#     )

#     st.markdown("---") # type: ignore
#     st.subheader("📊 System Status") # type: ignore

#     # Qdrant status
#     try:
#         info = qdrant.client.get_collection(qdrant.collection_name)
#         st.success(f"✅ Qdrant: {info.points_count} chunks indexed") # type: ignore
#     except Exception:
#         st.error("❌ Qdrant: Collection not found — run prepare_data.py first") # type: ignore

#     # Tavily rate limiter status
#     limiter = TavilyRateLimiter()
#     st.info(f"🌐 {limiter.get_status()}") # type: ignore

#     st.markdown("---") # type: ignore
#     st.caption(f"Domain: `{domain}`") # type: ignore

# # ── Main layout ───────────────────────────────────────────────────────────────
# main_col, trace_col = st.columns([2, 1])# type: ignore

# with main_col:
#     st.header("💬 Ask a Question")# type: ignore
#     query = st.text_area(# type: ignore
#         "Enter your question:",
#         placeholder="Type your question here...",
#         height=100
#     )
#     submit = st.button("🔍 Get Answer", use_container_width=True)# type: ignore

# with trace_col:
#     st.header("📊 Pipeline Trace")# type: ignore
#     trace_placeholder = st.empty()# type: ignore

# # ── Process query ─────────────────────────────────────────────────────────────
# if submit:
#     # ✅ Guard against empty query
#     if not query.strip():
#         st.warning("⚠️ Please enter a question before submitting.")# type: ignore
#         st.stop()# type: ignore

#     # Get domain-specific components
#     evaluator = get_evaluator(domain)
#     generator = get_generator(domain)

#     trace_steps = []

#     with st.spinner("Running CRAG pipeline..."):# type: ignore

#         # Step 1 — Retrieve
#         with st.spinner("Step 1: Retrieving documents..."): # type: ignore
#             query_emb     = embedder.embed_query(query)
#             retrieved_docs = qdrant.search(query_emb, top_k=5)    # uses query_points internally

#         trace_steps.append(f"✅ Step 1: Retrieved {len(retrieved_docs)} documents")
#         trace_placeholder.markdown("\n\n".join(trace_steps))

#         # Step 2 — Evaluate
#         with st.spinner("Step 2: Evaluating retrieval quality..."): # type: ignore
#             confidence, eval_details = evaluator.evaluate(query, retrieved_docs)

#         trace_steps.append(f"✅ Step 2: Confidence = {confidence:.2f}")
#         trace_placeholder.markdown("\n\n".join(trace_steps))

#         # Step 3 — Route
#         action = router.route(confidence, query, domain)
#         trace_steps.append(f"✅ Step 3: Action = `{action}`")
#         trace_placeholder.markdown("\n\n".join(trace_steps))

#         # Step 4 — Generate
#         if action == "REFUSE":
#             response = {
#                 "answer": (
#                     "I cannot reliably answer this question with the "     # ✅ no medical advice
#                     "available information. Please try rephrasing your "
#                     "question or consult an authoritative source."
#                 ),
#                 "confidence": confidence,
#                 "confidence_level": "low_confidence",
#                 "sources": []
#             }
#             trace_steps.append("❌ Step 4: Query refused — confidence too low")

#         elif action == "WEB_SEARCH":
#             with st.spinner("Step 4: Searching web for better sources..."): # type: ignore
#                 sources = fallback.handle_fallback(query, retrieved_docs, domain)
#                 context = "\n".join([
#                     s.get("text", s.get("content", ""))[:300]
#                     for s in sources[:3]
#                 ])
#                 # ✅ correct method name: generate_answer not generate
#                 response = generator.generate_answer(query, context, confidence, sources)
#             trace_steps.append("✅ Step 4: Web search fallback triggered")

#         else:   # GENERATE
#             with st.spinner("Step 4: Generating answer..."):
#                 context = "\n".join([d["text"] for d in retrieved_docs[:3]])
#                 # ✅ correct method name
#                 response = generator.generate_answer(
#                     query, context, confidence, retrieved_docs
#                 )
#             trace_steps.append("✅ Step 4: Answer generated from local knowledge base")

#         trace_placeholder.markdown("\n\n".join(trace_steps))

#     # ── Display answer ────────────────────────────────────────────────────────
#     st.divider() # type: ignore
#     st.header("💡 Answer") # type: ignore
#     st.write(response["answer"]) # type: ignore

#     # Confidence metrics
#     st.divider(),# type: ignore
#     # ✅ Use different variable names — col1/col2 already used above
#     m1, m2, m3 = st.columns(3) # type: ignore
#     with m1:
#         st.metric("Confidence Score", f"{response['confidence']:.0%}")# type: ignore
#     with m2:
#         st.metric("Sources Used", len(response.get("sources", []))) # type: ignore
#     with m3:
#         level = response.get("confidence_level", "unknown")
#         colour = {"high_confidence": "🟢", "medium_confidence": "🟡",
#                   "low_confidence": "🔴", "error": "❌"}.get(level, "⚪")
#         st.metric("Confidence Level", f"{colour} {level.replace('_', ' ').title()}") # type: ignore

#     # Source attribution
#     if response.get("sources"):
#         with st.expander("📚 View Sources"):# type: ignore
#             for i, source in enumerate(response["sources"][:5], 1):
#                 st.write(f"**Source {i}:** {source.get('title', 'Unknown')}")# type: ignore
#                 if source.get("url"):
#                     st.write(f"🔗 URL: {source['url']}")# type: ignore
#                 st.write(f"📁 Type: `{source.get('source_type', 'unknown')}`")# type: ignore
#                 if source.get("score"):
#                     st.write(f"⭐ Score: {source['score']:.3f}")# type: ignore
#                 st.divider() # type: ignore

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.divider()# type: ignore
# st.markdown( # type: ignore
#     "*CRAG — Generalised Corrective RAG with confidence-aware "   # ✅ no medical ref
#     "retrieval evaluation and web search fallback*"
# ) 

import streamlit as st
import sys
import tempfile
import os
sys.path.insert(0, '.')

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.core.VectorEmbeddings import GenerateEmbeddings
from src.Integrations.Qdrant_client import StoreQdrant
from src.core.evaluator import RetrievalEvaluator
from src.core.router import ConditionalRouer             
from src.core.fallback_handler import FallbackHandler
from src.core.AnswerGenerator import AnswerGenerator
from src.core.data_ingestion import DataIngestion          # ← fixed class name
from src.utils.tavily_rate_limiter import TavilyRateLimiter
from src.utils.logger import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRAG — Corrective RAG",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 CRAG — Corrective Retrieval Augmented Generation")
st.markdown("*Intelligent question answering with confidence scoring and web search fallback*")

# ── Cache base components ─────────────────────────────────────────────────────
@st.cache_resource
def init_base_components():
    embedder = GenerateEmbeddings()
    qdrant   = StoreQdrant()
    router   = ConditionalRouer({})
    fallback = FallbackHandler()
    ingestor = DataIngestion()          # ← fixed class name
    return embedder, qdrant, router, fallback, ingestor

# ✅ Initialize FIRST — before sidebar uses these variables
embedder, qdrant, router, fallback, ingestor = init_base_components()

# ── Cache evaluator + generator per domain ────────────────────────────────────
@st.cache_resource
def get_evaluator(domain: str) -> RetrievalEvaluator:
    return RetrievalEvaluator(domain=domain)

@st.cache_resource
def get_generator(domain: str) -> AnswerGenerator:
    return AnswerGenerator(domain=domain)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Document Upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to add to the knowledge base"
    )

    if uploaded_files:
        if st.button("⬆️ Ingest Documents", use_container_width=True):
            all_chunks = []
            all_metadata = []
            progress   = st.progress(0)
            status     = st.empty()

            for i, file in enumerate(uploaded_files):
                suffix = ".pdf" if file.type == "application/pdf" else ".txt"

                file_bytes = file.read()

                if not file_bytes:
                    st.warning(f"⚠️ {file.name} is empty — skipping")
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                status.text(f"Processing: {file.name}...")
                try:
                    # ✅ Load individual file directly
                    if suffix == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding="utf-8")

                    docs   = loader.load()
                    chunks = ingestor.split_data(docs)   # reuse split_data

                    for chunk in chunks:
                        all_metadata.append({
                            "title":       file.name,                    # real filename
                            "source":      file.name,
                            "source_type": "local_kb",
                            "page":        chunk.metadata.get("page", 0),
                        })

                    all_chunks.extend(chunks)
                    status.text(f"✅ {file.name}: {len(chunks)} chunks created")

                except Exception as e:
                    st.warning(f"⚠️ Failed to process {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

                progress.progress((i + 1) / len(uploaded_files))

            if all_chunks:
                status.text(f"Uploading {len(all_chunks)} chunks to Qdrant...")
                try:
                    chunk_texts = [c.page_content for c in all_chunks]
                    
                    # metadata = [
                    #     {
                    #         "title":       c.metadata.get("source", "unknown").split("/")[-1],
                    #         "source":      c.metadata.get("source", "unknown"),
                    #         "source_type": "local_kb",
                    #         "page":        c.metadata.get("page", 0),
                    #     }
                    #     for c in all_chunks
                    # ]

                    embeddings  = embedder.embed_chunks(chunk_texts)
                    qdrant.create_collection(force_recreate=False)
                    qdrant.upload_embeddings(
                        chunk_texts,
                        embeddings,
                        metadata=all_metadata    # ← real filenames passed here
                    )

                    # metadata = []
                    # file_index = 0
                    # chunks_per_file = []

                    # temp_chunks_count = 0
                    # for file in uploaded_files:
                    #     suffix = ".pdf" if file.type == "application/pdf" else ".txt"
                    #     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    #         tmp.write(file.read())
                    #         tmp_path = tmp.name
                    #     if suffix == ".pdf":
                    #         loader = PyPDFLoader(tmp_path)
                    #     else:
                    #         loader = TextLoader(tmp_path, encoding="utf-8")
                    #     file_docs = loader.load()
                    #     file_chunks = ingestor.split_data(file_docs)
                    #     chunks_per_file.append((file.name, len(file_chunks)))
                    #     os.unlink(tmp_path)

                    # # Build metadata matching chunk order
                    # for filename, count in chunks_per_file:
                    #     for _ in range(count):
                    #         metadata.append({
                    #             "title": filename,           
                    #             "source": filename,
                    #             "source_type": "local_kb",
                    #             "page": 0
                    #         })

                    # embeddings = embedder.embed_chunks(chunk_texts)
                    # qdrant.create_collection(force_recreate=False)
                    # qdrant.upload_embeddings(chunk_texts, embeddings, metadata=metadata)

                    st.success(
                        f"✅ {len(all_chunks)} chunks from "
                        f"{len(uploaded_files)} file(s) added to knowledge base"
                    )
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
            else:
                st.error("❌ No content extracted — check your files")

    st.markdown("---")
    st.header("⚙️ Configuration")
    domain = st.selectbox(
        "Select Domain",
        ["general", "legal", "technical", "financial", "academic"],
        index=0
    )

    st.markdown("---")
    st.subheader("📊 System Status")

    try:
        info = qdrant.client.get_collection(qdrant.collection_name)
        st.success(f"✅ Qdrant: {info.points_count} chunks indexed")
    except Exception:
        st.error("❌ Qdrant: Collection not found — run prepare_data.py first")

    limiter = TavilyRateLimiter()
    st.info(f"🌐 {limiter.get_status()}")

    st.markdown("---")
    st.caption(f"Domain: `{domain}`")

# ── Main layout ───────────────────────────────────────────────────────────────
main_col, trace_col = st.columns([2, 1])

with main_col:
    st.header("💬 Ask a Question")
    query = st.text_area(
        "Enter your question:",
        placeholder="Type your question here...",
        height=100
    )
    submit = st.button("🔍 Get Answer", use_container_width=True)

with trace_col:
    st.header("📊 Pipeline Trace")
    trace_placeholder = st.empty()

# ── Process query ─────────────────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
        st.stop()

    evaluator   = get_evaluator(domain)
    generator   = get_generator(domain)
    trace_steps = []

    with st.spinner("Running CRAG pipeline..."):

        # Step 1 — Retrieve
        with st.spinner("Step 1: Retrieving documents..."):
            query_emb      = embedder.embed_query(query)
            retrieved_docs = qdrant.search(query_emb, top_k=5)

        trace_steps.append(f"✅ Step 1: Retrieved {len(retrieved_docs)} documents")
        trace_placeholder.markdown("\n\n".join(trace_steps))

        # Step 2 — Evaluate
        with st.spinner("Step 2: Evaluating retrieval quality..."):
            confidence, eval_details = evaluator.evaluate(query, retrieved_docs)

        trace_steps.append(f"✅ Step 2: Confidence = {confidence:.2f}")
        trace_placeholder.markdown("\n\n".join(trace_steps))

        # Step 3 — Route
        action = router.route(confidence, query, domain)
        trace_steps.append(f"✅ Step 3: Action = `{action}`")
        trace_placeholder.markdown("\n\n".join(trace_steps))

        # Step 4 — Generate
        if action == "REFUSE":
            response = {
                "answer": (
                    "I cannot reliably answer this question with the "
                    "available information. Please try rephrasing your "
                    "question or consult an authoritative source."
                ),
                "confidence": confidence,
                "confidence_level": "low_confidence",
                "sources": []
            }
            trace_steps.append("❌ Step 4: Query refused — confidence too low")

        elif action == "WEB_SEARCH":
            with st.spinner("Step 4: Searching web for better sources..."):
                sources = fallback.handle_fallback(query, retrieved_docs, domain)
                context = "\n".join([
                    s.get("text", s.get("content", ""))[:300]
                    for s in sources[:3]
                ])
                response = generator.generate_answer(
                    query, context, confidence, sources
                )
            trace_steps.append("✅ Step 4: Web search fallback triggered")

        else:  # GENERATE
            with st.spinner("Step 4: Generating answer..."):
                context = "\n".join([d["text"] for d in retrieved_docs[:3]])
                response = generator.generate_answer(
                    query, context, confidence, retrieved_docs
                )
            trace_steps.append("✅ Step 4: Answer generated from local knowledge base")

        trace_placeholder.markdown("\n\n".join(trace_steps))

    # ── Display results ───────────────────────────────────────────────────────
    st.divider()
    st.header("💡 Answer")
    st.markdown(str(response["answer"]))    # ✅ fixed — no more Streamlit docs bug

    st.divider()                            # ✅ removed comma
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Confidence Score", f"{response['confidence']:.0%}")
    with m2:
        st.metric("Sources Used", len(response.get("sources", [])))
    with m3:
        level  = response.get("confidence_level", "unknown")
        colour = {
            "high_confidence":   "🟢",
            "medium_confidence": "🟡",
            "low_confidence":    "🔴",
            "error":             "❌"
        }.get(level, "⚪")
        st.metric("Confidence Level", f"{colour} {level.replace('_', ' ').title()}")

    if response.get("sources"):
        with st.expander("📚 View Sources"):
            for i, source in enumerate(response["sources"][:5], 1):

                title = (
                    source.get("title") or
                    source.get("source") or
                    source.get("url") or
                    "Local Document"
                )
                st.write(f"**Source {i}:** {title}")
                if source.get("url"):
                    st.write(f"🔗 URL: {source['url']}")
                st.write(f"📁 Type: `{source.get('source_type', 'unknown')}`")
                st.write(f"📄 Page: {source.get('page', 'N/A')}")

                if source.get("score"):
                    st.write(f"⭐ Score: {source['score']:.3f}")
                st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "*CRAG — Generalised Corrective RAG with confidence-aware "
    "retrieval evaluation and web search fallback*"
)