import streamlit as st
from src.pipeline import build_pipeline, ask

st.set_page_config(page_title="Insight Engine",page_icon="🔍")
st.title("🔍 Insight Engine - RAG")
st.caption("Ask any kind of questions related to Documents!!!")


@st.cache_resource
def load_pipeline():
    return build_pipeline("./data/text_files")

retriever, generator = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role":"user","content": question})

    with st.chat_message("assistant"):
        docs = retriever.retrieve(question, top_k=3)
        answer = generator.generate(question,docs)
        st.write(answer)

    st.session_state.messages.append({"role":"assistant", "content":answer})
    