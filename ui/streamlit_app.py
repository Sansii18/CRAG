import streamlit as st
import sys
sys.path.insert(0, '.')

from src.core.VectorEmbeddings import GenerateEmbeddings      # ← correct class name
from src.Integrations.Qdrant_client import StoreQdrant         # ← correct class name
from src.core.evaluator import RetrievalEvaluator
from src.core.router import ConditionalRouer
from src.core.fallback_handler import FallbackHandler
from src.core.AnswerGenerator import AnswerGenerator           # ← correct module name
from src.utils.tavily_rate_limiter import TavilyRateLimiter
from src.utils.logger import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config( # type: ignore
    page_title="AdaptCRAG — Corrective RAG",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AdaptCRAG — Corrective Retrieval Augmented Generation") # type: ignore
st.markdown("*Intelligent question answering with confidence scoring and web search fallback*") # type: ignore

# ── Cache heavy components that don't depend on domain ────────────────────────
@st.cache_resource # type: ignore
def init_base_components():
    """Initialise components that are domain-independent — cached once"""
    embedder = GenerateEmbeddings()
    qdrant   = StoreQdrant()
    router   = ConditionalRouer({})
    fallback = FallbackHandler()
    return embedder, qdrant, router, fallback


# ── Cache evaluator and generator per domain ─────────────────────────────────
@st.cache_resource # type: ignore
def get_evaluator(domain: str) -> RetrievalEvaluator:
    """Cached per domain — switching domain creates a new cached instance"""
    return RetrievalEvaluator(domain=domain)

@st.cache_resource # type: ignore
def get_generator(domain: str) -> AnswerGenerator:
    """Cached per domain"""
    return AnswerGenerator(domain=domain)

embedder, qdrant, router, fallback = init_base_components()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar: # type: ignore
    st.header("⚙️ Configuration") # type: ignore

    # ✅ medical removed, general + others added
    domain = st.selectbox( # type: ignore
        "Select Domain",
        ["general", "legal", "technical", "financial", "academic"],
        index=0
    )

    st.markdown("---") # type: ignore
    st.subheader("📊 System Status") # type: ignore

    # Qdrant status
    try:
        info = qdrant.client.get_collection(qdrant.collection_name)
        st.success(f"✅ Qdrant: {info.points_count} chunks indexed") # type: ignore
    except Exception:
        st.error("❌ Qdrant: Collection not found — run prepare_data.py first") # type: ignore

    # Tavily rate limiter status
    limiter = TavilyRateLimiter()
    st.info(f"🌐 {limiter.get_status()}") # type: ignore

    st.markdown("---") # type: ignore
    st.caption(f"Domain: `{domain}`") # type: ignore

# ── Main layout ───────────────────────────────────────────────────────────────
main_col, trace_col = st.columns([2, 1])# type: ignore

with main_col:
    st.header("💬 Ask a Question")# type: ignore
    query = st.text_area(# type: ignore
        "Enter your question:",
        placeholder="Type your question here...",
        height=100
    )
    submit = st.button("🔍 Get Answer", use_container_width=True)# type: ignore

with trace_col:
    st.header("📊 Pipeline Trace")# type: ignore
    trace_placeholder = st.empty()# type: ignore

# ── Process query ─────────────────────────────────────────────────────────────
if submit:
    # ✅ Guard against empty query
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")# type: ignore
        st.stop()# type: ignore

    # Get domain-specific components
    evaluator = get_evaluator(domain)
    generator = get_generator(domain)

    trace_steps = []

    with st.spinner("Running CRAG pipeline..."):# type: ignore

        # Step 1 — Retrieve
        with st.spinner("Step 1: Retrieving documents..."): # type: ignore
            query_emb     = embedder.embed_query(query)
            retrieved_docs = qdrant.search(query_emb, top_k=5)    # uses query_points internally

        trace_steps.append(f"✅ Step 1: Retrieved {len(retrieved_docs)} documents")
        trace_placeholder.markdown("\n\n".join(trace_steps))

        # Step 2 — Evaluate
        with st.spinner("Step 2: Evaluating retrieval quality..."): # type: ignore
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
                    "I cannot reliably answer this question with the "     # ✅ no medical advice
                    "available information. Please try rephrasing your "
                    "question or consult an authoritative source."
                ),
                "confidence": confidence,
                "confidence_level": "low_confidence",
                "sources": []
            }
            trace_steps.append("❌ Step 4: Query refused — confidence too low")

        elif action == "WEB_SEARCH":
            with st.spinner("Step 4: Searching web for better sources..."): # type: ignore
                sources = fallback.handle_fallback(query, retrieved_docs, domain)
                context = "\n".join([
                    s.get("text", s.get("content", ""))[:300]
                    for s in sources[:3]
                ])
                # ✅ correct method name: generate_answer not generate
                response = generator.generate_answer(query, context, confidence, sources)
            trace_steps.append("✅ Step 4: Web search fallback triggered")

        else:   # GENERATE
            with st.spinner("Step 4: Generating answer..."):
                context = "\n".join([d["text"] for d in retrieved_docs[:3]])
                # ✅ correct method name
                response = generator.generate_answer(
                    query, context, confidence, retrieved_docs
                )
            trace_steps.append("✅ Step 4: Answer generated from local knowledge base")

        trace_placeholder.markdown("\n\n".join(trace_steps))

    # ── Display answer ────────────────────────────────────────────────────────
    st.divider() # type: ignore
    st.header("💡 Answer") # type: ignore
    st.write(response["answer"]) # type: ignore

    # Confidence metrics
    st.divider(),# type: ignore
    # ✅ Use different variable names — col1/col2 already used above
    m1, m2, m3 = st.columns(3) # type: ignore
    with m1:
        st.metric("Confidence Score", f"{response['confidence']:.0%}")# type: ignore
    with m2:
        st.metric("Sources Used", len(response.get("sources", []))) # type: ignore
    with m3:
        level = response.get("confidence_level", "unknown")
        colour = {"high_confidence": "🟢", "medium_confidence": "🟡",
                  "low_confidence": "🔴", "error": "❌"}.get(level, "⚪")
        st.metric("Confidence Level", f"{colour} {level.replace('_', ' ').title()}") # type: ignore

    # Source attribution
    if response.get("sources"):
        with st.expander("📚 View Sources"):# type: ignore
            for i, source in enumerate(response["sources"][:5], 1):
                st.write(f"**Source {i}:** {source.get('title', 'Unknown')}")# type: ignore
                if source.get("url"):
                    st.write(f"🔗 URL: {source['url']}")# type: ignore
                st.write(f"📁 Type: `{source.get('source_type', 'unknown')}`")# type: ignore
                if source.get("score"):
                    st.write(f"⭐ Score: {source['score']:.3f}")# type: ignore
                st.divider() # type: ignore

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()# type: ignore
st.markdown( # type: ignore
    "*AdaptCRAG — Generalised Corrective RAG with confidence-aware "   # ✅ no medical ref
    "retrieval evaluation and web search fallback*"
) 