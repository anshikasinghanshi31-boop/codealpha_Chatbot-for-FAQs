"""Streamlit chat UI.  Run with:  streamlit run app.py"""
import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="College FAQ Bot", page_icon="🎓")
st.title("🎓 College FAQ Chatbot")
st.caption("Ask about admissions, fees, hostels, placements and more.")


@st.cache_resource
def load_bot():
    return FAQChatbot()


bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

with st.sidebar:
    st.header("Try asking")
    for item in bot.faqs[:8]:
        st.markdown(f"- {item['question']}")
    show_debug = st.toggle("Show match details", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    answer, matched, score, suggestions = bot.get_response(query)
    reply = answer
    if show_debug and matched:
        reply += f"\n\n*Matched: “{matched}” (similarity {score:.2f})*"
    if suggestions:
        reply += "\n\n**Related questions:**\n" + "\n".join(f"- {s}" for s in suggestions)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
