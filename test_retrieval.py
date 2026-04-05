
# --- test_retrieval_price_first.py ---
# Ưu tiên GIÁ (PRICE) khi retrieval & rerank
# - Retrieval: cố gắng lọc theo ngân sách (±30%) nếu có metadata PRICE_NUM. Nếu không, fallback không filter.
# - Rerank: bơm PRICE từ metadata vào văn bản doc + yêu cầu "ưu tiên giá".
# - Final score: ưu tiên giá với trọng số cao hơn độ liên quan ngôn ngữ.
#
# Yêu cầu cấu trúc metadata của Chroma chứa ít nhất 1 trong 2 khóa:
#   - PRICE_NUM (int, VND)  => khuyến nghị
#   - PRICE (str)           => sẽ được parse về số
#
# Thử nghiệm với SAMPLE_QUERY = "PC 30000000 gaming"
# Có thể thay đổi ALPHA_WEIGHT & PRICE_TOLERANCE tùy nhu cầu.

import os
import re
import math
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
DATA_DIR = "./data"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chromadb")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
SAMPLE_QUERY = "PC 50 triệu để làm việc"

# Ưu tiên giá: trọng số & dung sai
ALPHA_WEIGHT = 0.75           # 0.0..1.0  (tăng để ưu tiên giá mạnh hơn)
PRICE_TOLERANCE = 0.30        # ±30%      (điều chỉnh để siết/lỏng tiêu chí sát ngân sách)

# --- RERANK HELPERS (kế thừa từ bản gốc, có chỉnh nhẹ) ---
def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs_optimized(pairs, tokenizer, prefix, suffix, max_length):
    processed_pairs = []
    for pair_text in pairs:
        full_text = prefix + pair_text + suffix
        processed_pairs.append(full_text)

    inputs = tokenizer(
        processed_pairs,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

@torch.no_grad()
def compute_logits(model, inputs, token_true_id, token_false_id):
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]

    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# --- PRICE-FIRST UTILITIES ---
_VND_NUM_RE = re.compile(r'(?<!\d)(\d{1,3}(?:[._ ]?\d{3})+|\d{5,})(?!\d)')  # 30.000.000 | 30000000
_TRIEU_RE = re.compile(r'(?<!\w)(\d+(?:[.,]\d+)?)\s*(tr|triệu|m)(?!\w)', re.IGNORECASE)

def _digits_only(s: str) -> int | None:
    nums = re.findall(r'\d+', s or '')
    if not nums:
        return None
    return int(''.join(nums))

def parse_price_from_meta(meta) -> int | None:
    """Trả về giá (VND) từ metadata: ưu tiên PRICE_NUM, fallback parse PRICE (chuỗi)."""
    if not meta:
        return None
    if 'PRICE_NUM' in meta and isinstance(meta['PRICE_NUM'], (int, float)):
        return int(meta['PRICE_NUM'])
    val = meta.get('PRICE')
    if not val:
        return None
    # Thử các format phổ biến: '29.990.000', '30,000,000', '30 triệu', '30tr'
    m = _TRIEU_RE.search(str(val))
    if m:
        amt = float(m.group(1).replace(',', '.'))
        return int(round(amt * 1_000_000))
    n = _digits_only(str(val))
    if n and n >= 10_000:  # tránh nhầm số nhỏ
        return n
    return None

def extract_budget_vnd(text: str) -> int | None:
    """Tìm ngân sách từ query: hỗ trợ '30000000', '30.000.000', '30tr/triệu/m'..."""
    if not text:
        return None
    # Ưu tiên pattern '30tr', '30 triệu', '30m'
    m = _TRIEU_RE.search(text)
    if m:
        amt = float(m.group(1).replace(',', '.'))
        return int(round(amt * 1_000_000))
    # Tìm số dài => coi là VND
    m2 = _VND_NUM_RE.search(text.replace(',', '.').replace('đ', '').replace('vnd', '').lower())
    if m2:
        n = int(re.sub(r'[._ ]', '', m2.group(1)))
        return n if n >= 10_000 else None
    return None

def price_closeness(budget: int, price: int, tolerance: float = PRICE_TOLERANCE) -> float:
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

def rerank_documents(reranker_data, query, documents, task_instruction: str, batch_size: int = 4):
    model = reranker_data["model"]
    tokenizer = reranker_data["tokenizer"]

    rerank_pairs = [format_instruction(task_instruction, query, doc) for doc in documents]

    all_scores = []
    for i in tqdm(range(0, len(rerank_pairs), batch_size), desc="Reranking Batches"):
        batch_pairs = rerank_pairs[i:i + batch_size]

        inputs = process_inputs_optimized(
            batch_pairs, tokenizer,
            reranker_data["prefix"],
            reranker_data["suffix"],
            reranker_data["max_length"]
        )

        scores = compute_logits(
            model, inputs,
            reranker_data["token_true_id"],
            reranker_data["token_false_id"]
        )
        all_scores.extend(scores)

    return all_scores

def main():
    print("Loading models and setting up device...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, padding_side='left')

        # Dùng dtype thay vì torch_dtype như bản gốc đã sửa
        reranker_model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_NAME,
            dtype=torch.float16
        ).to(device).eval()

        print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load models. Error: {e}")
        return

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    reranker_data = {
        "model": reranker_model,
        "tokenizer": reranker_tokenizer,
        "token_false_id": reranker_tokenizer.convert_tokens_to_ids("no"),
        "token_true_id": reranker_tokenizer.convert_tokens_to_ids("yes"),
        "max_length": 512,
        "prefix": prefix,
        "suffix": suffix,
    }

    # --- Connect Chroma ---
    print(f"\nConnecting to ChromaDB at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    prebuilt_collection = chroma_client.get_collection(name="prebuilt_pcs")
    print("Successfully connected to ChromaDB.")

    # --- Build query embedding ---
    print(f"\nGenerating embedding for query: '{SAMPLE_QUERY}'")
    query_embedding = embedding_model.encode(SAMPLE_QUERY, prompt_name="query")

    # --- Extract budget & prepare filtered retrieval ---
    budget = extract_budget_vnd(SAMPLE_QUERY)
    if budget:
        print(f"Detected budget from query: {budget:,} VND (tolerance ±{int(PRICE_TOLERANCE*100)}%)")
    else:
        print("No budget detected from query. Proceeding without price filter.")

    num_candidates = 50
    print(f"Retrieving top {num_candidates} candidates from prebuilt collection...")

    results_prebuilt = None
    if budget:
        lower = int(budget * (1.0 - PRICE_TOLERANCE))
        upper = int(budget * (1.0 + PRICE_TOLERANCE))
        # Ưu tiên lọc theo PRICE_NUM nếu có
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

    # Fallback không filter
    if results_prebuilt is None:
        results_prebuilt = prebuilt_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=num_candidates
        )

    initial_documents = results_prebuilt['documents'][0]
    initial_metadatas = results_prebuilt['metadatas'][0]
    print(f"Found {len(initial_documents)} total candidates for reranking.")

    # --- Prepare augmented docs & price scores ---
    price_scores = []
    augmented_docs = []
    for doc, meta in zip(initial_documents, initial_metadatas):
        price = parse_price_from_meta(meta)
        augmented_docs.append(augment_doc_with_meta(doc, meta, price))
        if budget and price:
            price_scores.append(price_closeness(budget, price))
        elif budget and not price:
            # Không có giá => phạt nhẹ
            price_scores.append(0.1)
        else:
            # Không có ngân sách => điểm trung bình
            price_scores.append(0.5)

    # --- Build task instruction that PRIORITIZES PRICE ---
    if budget:
        task_instruction = (
            "Prioritize PRICE over use-case. Target budget is "
            f"{budget} VND with tolerance ±{int(PRICE_TOLERANCE*100)}%. "
            "Answer \"yes\" ONLY if the document's price appears to be within the tolerance; "
            "otherwise answer \"no\" even if the use-case seems relevant."
        )
    else:
        task_instruction = (
            "User did not specify a budget. Judge general relevance to the PC query."
        )

    # --- Rerank (language relevance) ---
    relevance_scores = rerank_documents(reranker_data, SAMPLE_QUERY, augmented_docs, task_instruction)

    # --- Combine scores: ưu tiên GIÁ ---
    final = []
    for meta, price_s, rel_s in zip(initial_metadatas, price_scores, relevance_scores):
        combined = ALPHA_WEIGHT * price_s + (1.0 - ALPHA_WEIGHT) * rel_s
        final.append((combined, price_s, rel_s, meta))

    final_sorted = sorted(final, key=lambda x: x[0], reverse=True)

    top_n = 5
    print(f"\n--- TOP {top_n} (PRICE-first) ---")
    for i, (combined, p_s, r_s, meta) in enumerate(final_sorted[:top_n], start=1):
        print(f"\n{i}. Final: {combined:.4f} | price_score: {p_s:.4f} | relevance: {r_s:.4f}")
        print(f"   Type: Pre-built PC")
        print(f"   Use Case: {meta.get('Nhu cầu', 'N/A')}")
        print(f"   Price: {meta.get('PRICE', meta.get('PRICE_NUM', 'N/A'))}")

if __name__ == "__main__":
    main()
