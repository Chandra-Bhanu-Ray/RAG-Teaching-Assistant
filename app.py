"""EduRAG - Streamlit UI.  Run:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

import config
from teaching_assistant import LEVELS, TeachingAssistant

st.set_page_config(page_title="EduRAG - AI Teaching Assistant", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_assistant() -> TeachingAssistant:
    return TeachingAssistant()


ta = get_assistant()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎓 EduRAG")
    st.caption(f"LLM provider: **{config.LLM_PROVIDER}**")
    files = st.file_uploader("Upload course material", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not files):
        for f in files:
            tmp = Path(tempfile.gettempdir()) / f.name
            tmp.write_bytes(f.getvalue())
            with st.spinner(f"Indexing {f.name}..."):
                n = ta.ingest(tmp)
            st.success(f"{f.name}: {n} chunks added" if n else f"{f.name}: already indexed")
    st.markdown("**Indexed documents**")
    for s in ta.store.sources() or ["(none yet)"]:
        st.write(f"- {s}")
    level = st.selectbox("Explanation level", list(LEVELS))
    if st.button("Reset knowledge base"):
        ta.reset()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

tab_chat, tab_quiz, tab_notes = st.tabs(["💬 Ask", "📝 Quiz", "📒 Revision notes"])


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**[{i}] {s['source']} - page {s['page']}** (similarity {s['score']:.2f})")
                st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


# ---------------- Chat ----------------
with tab_chat:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))
    if q := st.chat_input("Ask a question about your material..."):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = ta.ask(q, level=level, history=history)
                    st.markdown(res["answer"])
                    show_sources(res["sources"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": res["answer"], "sources": res["sources"]}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------- Quiz ----------------
with tab_quiz:
    topic = st.text_input("Quiz topic", key="quiz_topic")
    n_q = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate quiz", disabled=not topic):
        with st.spinner("Writing questions..."):
            try:
                st.session_state.quiz = ta.quiz(topic, n_q)
            except Exception as e:
                st.session_state.quiz = []
                st.error(f"Error: {e}")
        if not st.session_state.quiz:
            st.warning("Couldn't build a quiz - check your material or try another topic.")
    quiz = st.session_state.get("quiz", [])
    answers = {}
    for i, item in enumerate(quiz):
        answers[i] = st.radio(f"Q{i + 1}. {item['question']}", item["options"], index=None, key=f"q{i}")
    if quiz and st.button("Check answers"):
        score = 0
        for i, item in enumerate(quiz):
            correct = item["options"][item["answer_index"]]
            if answers[i] == correct:
                score += 1
                st.success(f"Q{i + 1}: Correct. {item['explanation']}")
            else:
                st.error(f"Q{i + 1}: Answer is **{correct}**. {item['explanation']}")
        st.metric("Score", f"{score} / {len(quiz)}")

# ---------------- Notes ----------------
with tab_notes:
    nt = st.text_input("Topic for revision notes", key="notes_topic")
    if st.button("Generate notes", disabled=not nt):
        with st.spinner("Summarising..."):
            try:
                res = ta.summarize(nt)
                st.markdown(res["answer"])
                show_sources(res["sources"])
            except Exception as e:
                st.error(f"Error: {e}")
