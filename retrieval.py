import torch
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- PRICE-FIRST UTILITIES ---
import re as _re_price

_VND_NUM_RE = _re_price.compile(r'(?<!\d)(\d{1,3}(?:[._ ]?\d{3})+|\d{5,})(?!\d)')  # 30.000.000 | 30000000
_TRIEU_RE = _re_price.compile(r'(?<!\w)(\d+(?:[.,]\d+)?)\s*(tr|triệu|m)(?!\w)', _re_price.IGNORECASE)

def _digits_only_price(s: str) -> int | None:
    nums = _re_price.findall(r'\d+', s or '')
    if not nums:
        return None
    try:
        return int(''.join(nums))
    except Exception:
        return None

def extract_budget_vnd(text: str) -> int | None:
    """Tìm ngân sách từ query: hỗ trợ '30000000', '30.000.000', '30tr/triệu/m'."""
    if not text:
        return None
    m = _TRIEU_RE.search(text)
    if m:
        try:
            amt = float(m.group(1).replace(',', '.'))
            return int(round(amt * 1_000_000))
        except Exception:
            pass
    cleaned = text.replace(',', '.').replace('đ', '').replace('vnd', '').lower()
    m2 = _VND_NUM_RE.search(cleaned)
    if m2:
        try:
            n = int(_re_price.sub(r'[._ ]', '', m2.group(1)))
            return n if n >= 10_000 else None
        except Exception:
            return None
    return None

def parse_price_from_meta(meta) -> int | None:
    """Trả về giá (VND) từ metadata: ưu tiên PRICE_NUM, fallback parse PRICE (chuỗi)."""
    if not meta:
        return None
    try:
        if 'PRICE_NUM' in meta and isinstance(meta['PRICE_NUM'], (int, float)):
            return int(meta['PRICE_NUM'])
    except Exception:
        pass
    val = (meta or {}).get('PRICE')
    if not val:
        return None
    m = _TRIEU_RE.search(str(val))
    if m:
        try:
            amt = float(m.group(1).replace(',', '.'))
            return int(round(amt * 1_000_000))
        except Exception:
            return None
    n = _digits_only_price(str(val))
    if n and n >= 10_000:
        return n
    return None

def price_closeness(budget: int, price: int, tolerance: float = 0.30) -> float:
    """Điểm 0..1: 1 khi sát ngân sách, giảm tuyến tính tới 0 ở |Δ| = tolerance*budget."""
    if not budget or not price:
        return 0.0
    diff = abs(price - budget) / max(1, budget)
    score = max(0.0, 1.0 - diff / max(1e-6, tolerance))
    return float(score)

def augment_doc_with_meta(doc: str, meta: dict, price: int | None) -> str:
    """Bơm metadata (PRICE, Nhu cầu) vào văn bản để reranker 'nhìn' được giá."""
    use_case = (meta or {}).get('Nhu cầu', 'N/A')
    price_str = f"{price}" if price is not None else str((meta or {}).get('PRICE', 'N/A'))
    header = f"[PRICE_VND={price_str}] [USE_CASE={use_case}]"
    return f"{header}\n{doc}"

# Tắt cảnh báo không cần thiết
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
DATA_DIR = "./data"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chromadb")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# --- HELPER FUNCTIONS (Internal to this module) ---

def _format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def _process_inputs_optimized(pairs, tokenizer, prefix, suffix, max_length):
    processed_pairs = [prefix + pair_text + suffix for pair_text in pairs]
    return tokenizer(
        processed_pairs,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

@torch.no_grad()
def _compute_logits(model, inputs, token_true_id, token_false_id):
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()

# --- PUBLIC FUNCTIONS (To be imported by app.py) ---

def setup_device():
    """Tự động nhận diện và trả về thiết bị tốt nhất có sẵn (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device

def load_embedding_model():
    """Tải và trả về mô hình embedding."""
    print("Loading embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

def load_reranker_data(device):
    """Tải mô hình reranker, tokenizer và các cấu hình cần thiết."""
    print("Loading reranker model and data...")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL_NAME, 
        dtype=torch.float16
    ).to(device).eval()
    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    return {
        "model": model, "tokenizer": tokenizer,
        "token_false_id": tokenizer.convert_tokens_to_ids("no"),
        "token_true_id": tokenizer.convert_tokens_to_ids("yes"),
        "max_length": 512, "prefix": prefix, "suffix": suffix,
    }

def rerank_documents(reranker_data, query, documents, batch_size=4, task_instruction=None):
    model = reranker_data["model"]
    tokenizer = reranker_data["tokenizer"]
    task_instruction = task_instruction or 'Given a user query about pre-built PCs, determine if the following document is relevant.'
    
    rerank_pairs = [_format_instruction(task_instruction, query, doc) for doc in documents]
    
    all_scores = []
    for i in tqdm(range(0, len(rerank_pairs), batch_size), desc="Reranking Batches"):
        batch_pairs = rerank_pairs[i:i + batch_size]
        inputs = _process_inputs_optimized(
            batch_pairs, tokenizer, 
            reranker_data["prefix"], 
            reranker_data["suffix"], 
            reranker_data["max_length"]
        )
        scores = _compute_logits(
            model, inputs, 
            reranker_data["token_true_id"], 
            reranker_data["token_false_id"]
        )
        all_scores.extend(scores)
        
    return all_scores

def perform_retrieval_and_reranking(query: str, embedding_model, reranker_data: dict):
    """
    Hàm chính thực hiện embedding → retrieval → reranking với ưu tiên GIÁ (PRICE).
    """
    # 1) Kết nối DB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    prebuilt_collection = chroma_client.get_collection(name="prebuilt_pcs")

    # 2) Tạo query embedding
    query_embedding = embedding_model.encode(query, prompt_name="query")

    # 3) Tách ngân sách từ query (nếu có) và retrieval ưu tiên giá
    budget = extract_budget_vnd(query)
    num_candidates = 50
    results_prebuilt = None
    if budget:
        lower = int(budget * 0.70)
        upper = int(budget * 1.30)
        try:
            results_prebuilt = prebuilt_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=num_candidates,
                where={"PRICE_NUM": {"$gte": lower, "$lte": upper}},
            )
            # Fallback nếu không có kết quả
            if not results_prebuilt or not results_prebuilt.get("documents") or not results_prebuilt["documents"][0]:
                results_prebuilt = None
        except Exception:
            results_prebuilt = None

    if results_prebuilt is None:
        results_prebuilt = prebuilt_collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=num_candidates
        )

    initial_documents = results_prebuilt['documents'][0]
    initial_metadatas = results_prebuilt['metadatas'][0]

    # 4) Chuẩn bị tài liệu cho rerank & điểm giá
    augmented_docs, price_scores = [], []
    for doc, meta in zip(initial_documents, initial_metadatas):
        p = parse_price_from_meta(meta)
        augmented_docs.append(augment_doc_with_meta(doc, meta, p))
        if budget and p:
            price_scores.append(price_closeness(budget, p, tolerance=0.30))
        elif budget and not p:
            price_scores.append(0.1)  # không có giá => phạt nhẹ
        else:
            price_scores.append(0.5)  # không có ngân sách => trung bình

    # 5) Rerank với instruction ưu tiên GIÁ
    if budget:
        instruction = (
            "Prioritize PRICE over use-case. Target budget is "
            f"{budget} VND with tolerance ±30%. "
            "Answer 'yes' ONLY if the document's price appears to be within the tolerance; "
            "otherwise answer 'no' even if the use-case seems relevant."
        )
    else:
        instruction = "User did not specify a budget. Judge general relevance to the PC query."

    relevance_scores = rerank_documents(reranker_data, query, augmented_docs, batch_size=4, task_instruction=instruction)

    # 6) Hợp nhất điểm với ưu tiên GIÁ
    ALPHA_WEIGHT = 0.75
    final = []
    for meta, price_s, rel_s in zip(initial_metadatas, price_scores, relevance_scores):
        combined = ALPHA_WEIGHT * price_s + (1.0 - ALPHA_WEIGHT) * rel_s
        final.append((combined, price_s, rel_s, meta))
    final_sorted = sorted(final, key=lambda x: x[0], reverse=True)

    # 7) Kết xuất top-k theo ngữ cảnh chuỗi để hiển thị
    top_n_reranked = 8
    reranked_context = []
    for i, (combined, p_s, r_s, meta) in enumerate(final_sorted[:top_n_reranked], start=1):
        link_value = meta.get('LINK') or meta.get('LINK SP')
        pc_details = [
            f"#{i} Final={combined:.3f} | price={p_s:.3f} | rel={r_s:.3f}",
            f"Prebuilt PC for {meta.get('Nhu cầu', 'N/A')}",
            f"CPU: {meta.get('Hãng CPU', '')} {meta.get('CPU', '')}",
            f"MAIN: {meta.get('MAIN', '')}",
            f"RAM: {meta.get('RAM', '')}",
            f"VGA: {meta.get('VGA', '')}",
            f"Storage: {meta.get('Storage', '')}",
            f"Price: {meta.get('PRICE', meta.get('PRICE_NUM', 'N/A'))}",
            f"Link: {link_value or 'N/A'}",
        ]
        reranked_context.append(" - ".join(filter(None, pc_details)))

    return "\\n".join(reranked_context) if reranked_context else "No relevant information found."


def perform_retrieval_wn(query: str, embedding_model):
    """
    Thực hiện retrieval cho Web Novels (Bỏ reranker, tối ưu truy xuất trực tiếp).
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        wn_collection = chroma_client.get_collection(name="web_novels")
    except ValueError:
        return "Web Novel collection is not initialized. Please run embedding script first."

    # Tạo query embedding
    query_embedding = embedding_model.encode(query, prompt_name="query")

    # Retrieval (Tự động trả về top các novels gần nhất)
    num_candidates = 5
    results = wn_collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=num_candidates
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant web novels found."

    wn_docs = results['documents'][0]
    wn_metas = results['metadatas'][0]

    # Xây dựng context
    retrieved_context = []
    for i, (doc, meta) in enumerate(zip(wn_docs, wn_metas), start=1):
        novel_details = [
            f"#{i} Web Novel",
            f"Title: {meta.get('title', 'N/A')}",
            f"Rating: {meta.get('rating', 'N/A')}",
            f"Chapters: {meta.get('chapters', 'N/A')}",
            f"Authors: {meta.get('authors', 'N/A')}",
            f"Language: {meta.get('language', 'N/A')}",
            f"URL: {meta.get('url', 'N/A')}",
            f"Details: {doc}"
        ]
        retrieved_context.append("\n - ".join(filter(None, novel_details)))

    return "\n\n".join(retrieved_context)