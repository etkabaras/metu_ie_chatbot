from __future__ import annotations

import traceback

import streamlit as st

from src.config import Settings
from src.models import ChatTurn
from src.service import ChatbotService


st.set_page_config(page_title="METU IE Summer Practice Chatbot", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def load_service() -> ChatbotService:
    settings = Settings()
    return ChatbotService.from_disk(settings)


def reset_service_cache() -> None:
    load_service.clear()


def render_source_card(index: int, source_item) -> None:
    source = source_item.chunk
    with st.expander(f"Source {index}: {source.title}", expanded=False):
        st.markdown(f"**Source URL / Path:** `{source.source}`")
        st.markdown(f"**Official METU source:** `{'yes' if source.official else 'no'}`")
        st.markdown(f"**Retrieval score:** `{source_item.score:.3f}`")
        st.write(source.content)


def main() -> None:
    st.title("METU IE Summer Practice Chatbot")
    st.caption("Answers only from official METU sources plus your configured local context.")

    settings = Settings()

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Provider:** `{settings.llm_provider}`")
        if settings.llm_provider == "gemini":
            st.write(f"**Model:** `{settings.gemini_model}`")
        elif settings.llm_provider == "openai":
            st.write(f"**Model:** `{settings.openai_model}`")
        st.write(f"**Index path:** `{settings.index_path}`")

        if st.button("Rebuild knowledge index", use_container_width=True):
            try:
                reset_service_cache()
                service = load_service()
                service.rebuild_index()
                reset_service_cache()
                st.success("Knowledge index rebuilt.")
            except Exception as exc:
                st.error(f"Could not rebuild index: {exc}")

        st.divider()
        st.markdown(
            """
**Persistent context folders**
- `data/custom_context/always_on/`
- `data/custom_context/user_docs/`

After changing those files, rebuild the index.
"""
        )

        st.divider()
        st.markdown(
            """
**Good test questions**
- What are the requirements for IE300?
- What should I do before summer practice?
- Where can I find the IE400 application form?
- What does the insurance page say?
"""
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for idx, source in enumerate(message.get("sources", []), start=1):
                render_source_card(idx, source)

    query = st.chat_input("Ask a METU IE Summer Practice question...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    st.session_state.history.append(ChatTurn(role="user", content=query))
    with st.chat_message("user"):
        st.markdown(query)

    try:
        service = load_service()
        with st.chat_message("assistant"):
            with st.spinner("Checking the approved sources..."):
                result = service.answer(query=query, history=st.session_state.history)

            st.markdown(result.answer)
            if result.sources:
                st.markdown("**Sources used**")
                for idx, source in enumerate(result.sources, start=1):
                    render_source_card(idx, source)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )
        st.session_state.history.append(ChatTurn(role="assistant", content=result.answer))
    except Exception as exc:
        error_text = f"Application error: {exc}"
        with st.chat_message("assistant"):
            st.error(error_text)
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
        st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": []})


if __name__ == "__main__":
    main()
