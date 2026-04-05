import streamlit as st
import openai
import re
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import các hàm cần thiết từ module retrieval của chúng ta
import retrieval as retrieval
# --- MODEL LAZY-LOAD / OFFLOAD HELPERS ---
import gc
import torch

def _ensure_models(mode="PC"):
    """Chỉ load mô hình khi cần. Lưu vào session_state để tái sử dụng trong phiên."""
    if "device" not in st.session_state:
        st.session_state.device = cached_setup_device()
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = cached_load_embedding_model()
    if mode == "PC" and "reranker_data" not in st.session_state:
        st.session_state.reranker_data = cached_load_reranker_data(st.session_state.device)

def _unload_models():
    """Offload mô hình khỏi GPU/CPU và giải phóng bộ nhớ."""
    # Reranker
    if "reranker_data" in st.session_state:
        rd = st.session_state.pop("reranker_data")
        try:
            model = rd.get("model")
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                del model
        except Exception:
            pass
        try:
            tok = rd.get("tokenizer")
            del tok
        except Exception:
            pass
        del rd

    # Embedding
    if "embedding_model" in st.session_state:
        em = st.session_state.pop("embedding_model")
        try:
            # SentenceTransformer thường có .to("cpu")
            if hasattr(em, "to"):
                em.to("cpu")
        except Exception:
            pass
        del em

    # Không xóa device để code khác còn đọc device hiện tại
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()

# --- END HELPERS ---

# --- CONFIGURATION ---
st.set_page_config(page_title="PC Assistant Chatbot", layout="wide")

# --- UTILITY FUNCTIONS (For UI and OpenAI) ---

def parse_response(response: str):
    """Tách phần suy nghĩ của LLM ra khỏi câu trả lời cuối cùng."""
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        thought_content = match.group(1).strip()
        clean_response = re.sub(think_pattern, '', response, count=1, flags=re.DOTALL).strip()
        return thought_content, clean_response
    return None, response.strip()

def extract_purchase_links(context: str):
    """Lấy danh sách liên kết mua từ chuỗi context."""
    if not context:
        return []
    matches = re.findall(r"Link:\s*(https?://\S+)", context)
    cleaned, seen = [], set()
    for link in matches:
        normalized = link.rstrip('.,)')
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned

def get_openai_client(api_key: str, base_url: str):
    """Khởi tạo OpenAI client."""
    if not api_key or not base_url: return None
    try:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

def generate_response_stream(client: openai.OpenAI, messages: list, retrieved_info: str, model: str):
    """Yields response chunks from the LLM API stream."""
    if not client:
        yield "Error: OpenAI client not initialized."
        return

    try:
        final_messages = build_generation_messages(messages, retrieved_info)
        stream = client.chat.completions.create(
            model=model,
            messages=final_messages,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except openai.APIError as e:
        yield f"Error generating response: {e}"

def generate_response_local(messages: list, retrieved_info: str, model_name: str, max_new_tokens: int = 1024):
    try:
        tokenizer, model = cached_load_local_fallback(model_name)
        final_messages = build_generation_messages(messages, retrieved_info)
        text = tokenizer.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
        output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1]:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        thought_content, clean_response = parse_response(content)
        return thought_content or "", clean_response or content, None
    except Exception as err:
        return None, None, str(err)

# --- MODEL LOADING (with Streamlit Caching) ---

@st.cache_resource(show_spinner="Setting up device...")
def cached_setup_device():
    return retrieval.setup_device()

@st.cache_resource(show_spinner="Loading embedding model (Qwen-Embedding)...")
def cached_load_embedding_model():
    return retrieval.load_embedding_model()

@st.cache_resource(show_spinner="Loading reranker model (Qwen-Reranker)...")
def cached_load_reranker_data(device):
    return retrieval.load_reranker_data(device)

@st.cache_resource(show_spinner="Loading local fallback model...")
def cached_load_local_fallback(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

# --- STREAMLIT UI ---
st.title("PC Assistant Chatbot 💬")
# (lazy) models will be loaded on demand via _ensure_models()
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    st.info("Configure the API endpoint for your **generation** LLM below.")
    generation_api = st.text_input("API Endpoint Base URL", value="http://127.0.0.1:1234/v1")
    api_key = st.text_input("API Key", type="password", value="not-needed")
    client = get_openai_client(api_key, generation_api) if 'get_openai_client' in globals() else None
    selected_model = st.text_input("Select Generation Model", value="qwen/qwen3-4b-2507")
    fallback_enabled = st.checkbox("Enable local fallback", value=True)
    fallback_model_name = st.text_input("Fallback HF Model", value="Qwen/Qwen3-4B-Instruct-2507")
    st.divider()
    st.subheader("Domain Selection")
    chatbot_mode = st.radio("Chế độ tư vấn", ["PC Advisor", "Web Novel"], index=0)
    st.divider()
    st.subheader("🧠 Model session")
    auto_offload = st.checkbox("Auto offload after each answer", value=False, help="Tự động giải phóng mô hình sau khi trả lời xong. Bật khi RAM/GPU hạn chế.")
    if st.button("Unload models now"):
        _unload_models()
        st.success("Models unloaded from memory.")
if "messages" not in st.session_state:
    st.session_state.messages = []

# <<< THAY ĐỔI 2: CẬP NHẬT GIAO DIỆN LỊCH SỬ TRÒ CHUYỆN >>>
# Sử dụng st.expander để hiển thị khối <think> trong lịch sử.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "thoughts" in message and message["thoughts"]:
            with st.expander("Show Thought Process"):
                st.markdown(message["thoughts"])
        st.markdown(message["content"])

def build_generation_messages(messages, retrieved_info, sys_prompt):
    context_message = {
        "role": "system",
        "content": f"Retrieved Information:\n{retrieved_info.strip() if retrieved_info else 'No retrieved context available.'}"
    }
    return [{"role": "system", "content": sys_prompt}, context_message] + messages

SYSTEM_PROMPT = (
    "You are an expert AI assistant that rewrites user queries for a vector database search "
    "and generates recommendations based on retrieved results.\n\n"
    "The vector database contains ONLY **pre-built / assembled PC systems**, each including:\n"
    "- Full hardware specifications (CPU, GPU, RAM, SSD, PSU, Case...)\n"
    "- Intended use tags (gaming, office, editing, rendering...)\n"
    "- Price in VND\n"
    "- A purchase URL\n\n"
    "Your tasks:\n"
    "1. Analyze and rewrite the user's natural-language query into a concise, keyword-rich search query.\n"
    "2. Run retrieval on pre-built PCs ONLY.\n"
    "3. Select **2–3 PCs** that best match the user's intent.\n"
    "4. For each recommended PC, ALWAYS include:\n"
    "   - Key specs (Motherboard,CPU, GPU, RAM, SSD...)\n"
    "   - Price (VND)\n"
    "   - The exact purchase link\n\n"
    "Note: User can pay more or less depending on their needs, not the exact budget mentioned.\n\n"
    "Guidelines for rewriting the query:\n"
    "- If the user mentions a budget, include it in the rewritten query.\n"
    "- If the user mentions usage (gaming, văn phòng, đồ họa, render, học tập…), include it.\n"
    "- Expand vague descriptions into exact intent (e.g. 'máy mạnh' → 'PC gaming mạnh hiệu năng cao').\n"
    "- Use both Vietnamese + English keywords if useful.\n"
    "- Answer in Vietnamese.\n"
    "- All currency is VND.\n\n"
    "Guidelines for generating recommendations:\n"
    "- if possible, output 2–3 options.\n"
    "- MUST use products retrieved from the vector database.\n"
    "- Each option must contain an accurate purchase link.\n"
    "- If the user gives vague input, infer the most reasonable use-case.\n\n"
    "- Parts listed are 2025 newest models available in the market.\n"
)

SYSTEM_PROMPT_WN = (
    "You are an expert AI assistant that recommends Web Novels to users based on retrieved data.\n\n"
    "The vector database contains pre-processed **Web Novels**, each including:\n"
    "- Title, Author, Genres, Tags, Description, Language\n"
    "- Rating and Chapters\n"
    "- A URL to read/view the novel\n\n"
    "Your tasks:\n"
    "1. Recommend 2 to 5 web novels that best match the user's intent from the retrieved context.\n"
    "2. For each recommendation, ALWAYS include:\n"
    "   - Title, Genres/Tags, and Author\n"
    "   - A short explanation of why it fits the user's request based on the description.\n"
    "   - The exact URL to the novel.\n\n"
    "3. Be friendly and conversational.\n"
    "4. Answer in Vietnamese.\n"
)

if prompt := st.chat_input("Nhập câu hỏi của bạn... / Hỏi về linh kiện hoặc Web Novel..."):
    start_time = time.time()  # Start measuring total latency
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và xếp hạng thông tin..."):
            if chatbot_mode == "PC Advisor":
                _ensure_models(mode="PC")
                embedding_model = st.session_state.embedding_model
                reranker_data = st.session_state.reranker_data
                retrieved_info = retrieval.perform_retrieval_and_reranking(
                    prompt,
                    embedding_model,
                    reranker_data
                )
                current_sys_prompt = SYSTEM_PROMPT
            else:
                _ensure_models(mode="WN")
                embedding_model = st.session_state.embedding_model
                retrieved_info = retrieval.perform_retrieval_wn(
                    prompt,
                    embedding_model
                )
                current_sys_prompt = SYSTEM_PROMPT_WN
            with st.expander("Show Reranked Context"):
                st.info(retrieved_info or "No context found.")
            if auto_offload:
                _unload_models()
        answer_placeholder = st.empty()
        thought_expander = st.expander("Show Thought Process")
        final_response_with_links = ""
        full_raw_response = ""
        thought_content = ""
        clean_response = ""
        is_thinking_parsed = False
        generation_success = False

        if client:
            try:
                final_messages = build_generation_messages(st.session_state.messages, retrieved_info, sys_prompt=current_sys_prompt)
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=final_messages,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_raw_response += content

                    if "</think>" in full_raw_response and not is_thinking_parsed:
                        temp_thought, temp_clean = parse_response(full_raw_response)
                        if temp_thought:
                            thought_content = temp_thought
                            clean_response = temp_clean

                            # Điền nội dung vào expander đã tạo
                            thought_expander.markdown(thought_content)

                            answer_placeholder.markdown(clean_response)
                            is_thinking_parsed = True
                    elif is_thinking_parsed:
                        clean_response += chunk
                        answer_placeholder.markdown(clean_response)
                    else:
                        answer_placeholder.markdown(full_raw_response)

                _, final_clean_response = parse_response(full_raw_response)
                final_clean_response = (final_clean_response or "").strip()
                purchase_links = extract_purchase_links(retrieved_info)
                final_response_with_links = final_clean_response
                answer_placeholder.markdown(final_response_with_links or full_raw_response)
                generation_success = True
            except Exception as err:
                st.warning(f"Remote generation failed, switching to fallback if available: {err}")
        else:
            st.warning("Cannot generate response. LLM API client not configured.")

        if not generation_success and fallback_enabled:
            with st.spinner("Đang sử dụng mô hình dự phòng..."):
                try:
                    tokenizer, local_model = cached_load_local_fallback(fallback_model_name)
                    final_messages = build_generation_messages(st.session_state.messages, retrieved_info, sys_prompt=current_sys_prompt)
                    text = tokenizer.apply_chat_template(
                        final_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    model_inputs = tokenizer([text], return_tensors="pt")
                    model_inputs = {k: v.to(local_model.device) for k, v in model_inputs.items()}
                    generated_ids = local_model.generate(
                        **model_inputs,
                        max_new_tokens=1024,
                        temperature=0.7,
                    )
                    output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1]:].tolist()
                    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    local_thought, local_response = parse_response(content)
                    fallback_error = None
                except Exception as err:
                    local_response = None
                    fallback_error = str(err)
                    local_thought = None
            if local_response:
                if local_thought:
                    thought_expander.markdown(local_thought)
                answer_placeholder.markdown(local_response)
                final_response_with_links = local_response
                thought_content = local_thought or ""
                generation_success = True
            else:
                st.error(f"Local fallback failed: {fallback_error}")
        elif not generation_success:
            st.error("Không thể tạo câu trả lời vì không có mô hình khả dụng.")

        if generation_success and final_response_with_links:
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response_with_links,
                "thoughts": thought_content
            })

    # Calculate and display total latency
    end_time = time.time()
    total_latency = end_time - start_time
    run_time_seconds = round(total_latency, 2)
    st.info(f"Thời gian xử lý: {run_time_seconds} giây")
    
    # Feedback form
    with st.form(key=f"feedback_{len(st.session_state.messages)}"):
        st.subheader("Phản hồi về câu trả lời")
        accuracy = st.radio(
            "Độ chính xác của cấu hình tư vấn:",
            ["correct_or_acceptable", "incorrect_or_not_suitable"],
            index=0,
            help="Chọn 'correct_or_acceptable' nếu cấu hình đúng/chấp nhận được, ngược lại chọn 'incorrect_or_not_suitable'."
        )
        context_score = st.slider(
            "Điểm hiểu ngữ cảnh (1-5):",
            min_value=1,
            max_value=5,
            value=3,
            help="1: hiểu sai hoàn toàn, 5: hiểu đúng và phản hồi hợp lý"
        )
        personalization_score = st.slider(
            "Điểm tính cá nhân hóa (1-5):",
            min_value=1,
            max_value=5,
            value=3,
            help="Mức độ phù hợp với sở thích, ngân sách, ràng buộc cá nhân"
        )
        submitted = st.form_submit_button("Gửi phản hồi")
        
        if submitted:
            FEEDBACK_CSV_PATH = Path(__file__).resolve().parent / "feedback.csv"
            try:
                file_exists = FEEDBACK_CSV_PATH.exists()
                with FEEDBACK_CSV_PATH.open(mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["timestamp", "question", "answer", "accuracy", "context_score", "personalization_score", "run_time_seconds"])
                    writer.writerow([
                        datetime.now().isoformat(),
                        prompt,
                        final_response_with_links,
                        accuracy,
                        context_score,
                        personalization_score,
                        run_time_seconds
                    ])
                st.success("Cảm ơn phản hồi của bạn!")
            except OSError as err:
                st.error(f"Không thể ghi feedback.csv: {err}")