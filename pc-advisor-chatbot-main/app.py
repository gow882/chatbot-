import streamlit as st
import openai
import re
import time
import csv
from datetime import datetime
from pathlib import Path
import retrieval as retrieval

# ================= CONFIG =================
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot 💬")

# ================= UTILS =================
def parse_response(response: str):
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        clean = re.sub(think_pattern, '', response, count=1, flags=re.DOTALL).strip()
        return thought, clean
    return None, response.strip()

def get_openai_client(api_key: str, base_url: str):
    try:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    except:
        return None

def build_messages(messages, retrieved_info, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{retrieved_info}"}
    ] + messages

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Config")

    generation_api = st.text_input(
        "API URL",
        value="https://openrouter.ai/api/v1"  # ✅ đổi sang API public
    )

    api_key = st.text_input("API Key", type="password")

    model = st.text_input("Model", value="openai/gpt-3.5-turbo")

    mode = st.radio("Mode", ["PC Advisor", "Web Novel"])

client = get_openai_client(api_key, generation_api)

# ================= CHAT HISTORY =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= PROMPT =================
SYSTEM_PC = "Bạn là AI tư vấn PC. Trả lời bằng tiếng Việt."
SYSTEM_WN = "Bạn là AI gợi ý truyện. Trả lời bằng tiếng Việt."

# ================= CHAT =================
if prompt := st.chat_input("Nhập câu hỏi..."):
    start = time.time()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):

            # 👉 retrieval nhẹ
            if mode == "PC Advisor":
                retrieved = retrieval.perform_retrieval(prompt)
                sys_prompt = SYSTEM_PC
            else:
                retrieved = retrieval.perform_retrieval_wn(prompt)
                sys_prompt = SYSTEM_WN

            full_text = ""

            if client:
                try:
                    messages = build_messages(st.session_state.messages, retrieved, sys_prompt)

                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True
                    )

                    placeholder = st.empty()

                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_text += content
                            placeholder.markdown(full_text)

                except Exception as e:
                    full_text = f"Lỗi API: {e}"
                    st.error(full_text)
            else:
                full_text = "Chưa cấu hình API"
                st.warning(full_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text
    })

    # thời gian xử lý
    st.info(f"⏱ {round(time.time() - start, 2)}s")

    # lưu feedback
    with st.form("feedback"):
        st.subheader("Feedback")
        score = st.slider("Điểm", 1, 5, 3)
        submit = st.form_submit_button("Gửi")

        if submit:
            path = Path("feedback.csv")
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), prompt, full_text, score])
            st.success("Đã lưu feedback")