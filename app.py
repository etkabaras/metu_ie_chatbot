import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from datetime import datetime

# ── Ayarlar ──────────────────────────────────────────────
LLM_MODEL = "gemini-2.0-flash"
TOP_K = 3
BASE_URL = "https://sp-ie.metu.edu.tr"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

PAGES = [
    "/en",
    "/en/general-information",
    "/en/steps-follow",
    "/en/forms",
    "/en/faq",
    "/en/sp-committee",
    "/en/sp-opportunities",
]

# ── Sayfa ayarları ───────────────────────────────────────
st.set_page_config(page_title="METU IE Chatbot", page_icon="🎓")

# ── Gemini API key ───────────────────────────────────────
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]


# ══════════════════════════════════════════════════════════
#  GEMINI REST API
# ══════════════════════════════════════════════════════════

def gemini_embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Gemini REST API ile embedding oluşturur."""
    url = f"{GEMINI_BASE}/models/text-embedding-004:embedContent?key={GEMINI_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def gemini_chat(system_prompt: str, messages: list) -> str:
    """Gemini REST API ile chat yapar."""
    url = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GEMINI_KEY}"

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2}
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


# ══════════════════════════════════════════════════════════
#  SCRAPING
# ══════════════════════════════════════════════════════════

def verify_metu_login(username: str, password: str) -> tuple[bool, str]:
    try:
        resp = requests.get(BASE_URL + "/en", auth=(username, password), timeout=10)
        if resp.status_code == 200:
            return True, "Giriş başarılı!"
        elif resp.status_code == 401:
            return False, "Kullanıcı adı veya şifre hatalı."
        else:
            return False, f"Sunucu hatası: {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Siteye bağlanılamadı. İnternet bağlantınızı kontrol edin."
    except requests.exceptions.Timeout:
        return False, "Bağlantı zaman aşımına uğradı."
    except Exception as e:
        return False, f"Beklenmeyen hata: {e}"


def scrape_site(username: str, password: str) -> list[dict]:
    session = requests.Session()
    session.auth = (username, password)
    all_content = []

    for page in PAGES:
        url = BASE_URL + page
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            content_div = (
                soup.find("div", class_="region-content")
                or soup.find("main")
                or soup.find("article")
            )

            if content_div:
                for tag in content_div.find_all(["nav", "header", "footer", "script", "style"]):
                    tag.decompose()
                text = content_div.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            clean_text = "\n".join(lines)

            title_tag = soup.find("h2")
            page_title = title_tag.get_text(strip=True) if title_tag else page.split("/")[-1].replace("-", " ").title()

            all_content.append({
                "title": page_title,
                "url": url,
                "content": clean_text,
            })
        except Exception:
            pass

    return all_content


# ══════════════════════════════════════════════════════════
#  CHUNKING & IN-MEMORY VECTOR STORE
# ══════════════════════════════════════════════════════════

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Sayfaları anlamlı chunk'lara ayırır."""
    chunks = []

    for page in pages:
        lines = page["content"].split("\n")
        page_title = page["title"]
        current_chunk = []
        current_title = page_title

        for line in lines:
            stripped = line.strip()
            is_heading = False
            if stripped and len(stripped) < 80:
                if stripped.endswith(":") and not stripped.startswith("-") and not stripped.startswith("*"):
                    if stripped[0].isalpha():
                        is_heading = True

            if is_heading and current_chunk and len("\n".join(current_chunk)) > 100:
                chunks.append({
                    "title": current_title,
                    "content": "\n".join(current_chunk).strip()
                })
                current_title = f"{page_title} > {stripped.rstrip(':')}"
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            content = "\n".join(current_chunk).strip()
            if len(content) > 50:
                chunks.append({"title": current_title, "content": content})

    return chunks


def build_vector_store(chunks: list[dict]) -> dict:
    """Chunk'ları embed edip bellekte saklar."""
    texts = [c["content"] for c in chunks]
    titles = [c["title"] for c in chunks]

    embeddings = []
    for text in texts:
        emb = gemini_embed(text, "RETRIEVAL_DOCUMENT")
        embeddings.append(emb)

    return {
        "texts": texts,
        "titles": titles,
        "embeddings": np.array(embeddings)
    }


def retrieve_context(store: dict, query: str, top_k: int = TOP_K) -> str:
    """Cosine similarity ile en yakın chunk'ları getirir."""
    query_emb = np.array(gemini_embed(query, "RETRIEVAL_QUERY"))

    # Cosine similarity hesapla
    embeddings = store["embeddings"]
    dot_products = embeddings @ query_emb
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    similarities = dot_products / norms

    # En yüksek benzerlik skorlarını bul
    top_indices = np.argsort(similarities)[::-1][:top_k]

    context_parts = []
    for idx in top_indices:
        sim = similarities[idx]
        context_parts.append(
            f"[{store['titles'][idx]}] (benzerlik: {sim:.2f})\n{store['texts'][idx]}"
        )

    return "\n\n---\n\n".join(context_parts)


# ══════════════════════════════════════════════════════════
#  ANA UYGULAMA
# ══════════════════════════════════════════════════════════

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None


# ── LOGIN SAYFASI ────────────────────────────────────────
if not st.session_state.authenticated:
    st.title("🎓 METU IE Summer Practice Chatbot")
    st.markdown("---")
    st.markdown("### 🔐 METU Girişi")
    st.caption("sp-ie.metu.edu.tr bilgilerinizle giriş yapın")

    with st.form("login_form"):
        username = st.text_input("Kullanıcı Adı", placeholder="Username")
        password = st.text_input("Şifre", type="password", placeholder="Password")
        submitted = st.form_submit_button("Giriş Yap", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Kullanıcı adı ve şifre boş bırakılamaz.")
        else:
            with st.spinner("Giriş yapılıyor..."):
                success, message = verify_metu_login(username, password)

            if success:
                st.session_state.authenticated = True
                st.session_state.metu_user = username
                st.session_state.metu_pass = password

                with st.spinner("📡 Site taranıyor ve veritabanı oluşturuluyor..."):
                    pages = scrape_site(username, password)

                    if pages:
                        chunks = chunk_pages(pages)
                        store = build_vector_store(chunks)
                        st.session_state.store = store
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.page_count = len(pages)

                st.rerun()
            else:
                st.error(message)

    st.stop()


# ── CHATBOT SAYFASI ──────────────────────────────────────
st.title("🎓 METU IE Summer Practice Chatbot")
st.caption("RAG-powered • Google Gemini • Siteden güncel veri çeker")

if st.session_state.store is None:
    st.error("Veritabanı bulunamadı. Lütfen tekrar giriş yapın.")
    st.session_state.authenticated = False
    st.rerun()

store = st.session_state.store

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Sorunuzu yazın...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("İlgili bilgiler aranıyor..."):
        context = retrieve_context(store, user_input)

    system_prompt = f"""You are a chatbot for METU Industrial Engineering Summer Practice.

ONLY use the context below to answer. This context was retrieved based on the user's question.
If the answer is not in the context, say:
"I could not find this information in the official dataset."

Answer clearly and shortly.

RELEVANT CONTEXT:
{context}
"""

    try:
        reply = gemini_chat(system_prompt, st.session_state.messages)
    except Exception as e:
        reply = f"Hata: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.get('metu_user', '')}")
    st.markdown("---")
    st.markdown("### ⚙️ Sistem Bilgisi")
    st.markdown(f"- **Embedding:** `text-embedding-004`")
    st.markdown(f"- **LLM:** `{LLM_MODEL}`")
    st.markdown(f"- **Top-K:** `{TOP_K}` chunk")
    st.markdown(f"- **Vektör DB:** In-memory (NumPy)")
    st.markdown(f"- **Taranan sayfa:** `{st.session_state.get('page_count', '?')}`")
    st.markdown(f"- **Toplam chunk:** `{st.session_state.get('chunk_count', '?')}`")
    st.markdown("---")

    if st.button("🔄 Veriyi Güncelle", use_container_width=True):
        with st.spinner("📡 Site yeniden taranıyor..."):
            pages = scrape_site(
                st.session_state.metu_user,
                st.session_state.metu_pass
            )
            if pages:
                chunks = chunk_pages(pages)
                store = build_vector_store(chunks)
                st.session_state.store = store
                st.session_state.chunk_count = len(chunks)
                st.session_state.page_count = len(pages)
                st.success("✅ Veri güncellendi!")

    if st.button("🚪 Çıkış Yap", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("📧 ie-staj@metu.edu.tr")
    st.markdown("🌐 [sp-ie.metu.edu.tr](http://sp-ie.metu.edu.tr)")
