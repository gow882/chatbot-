# --- test_retrieval.py ---

import torch
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
DATA_DIR = "./data"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chromadb")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
SAMPLE_QUERY = "PC 30000000 gaming"

# --- HELPER FUNCTIONS FOR RERANKING ---

def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# SỬA LỖI 1: Thêm `prefix` và `suffix` làm tham số cho hàm
def process_inputs_optimized(pairs, tokenizer, prefix, suffix, max_length):
    processed_pairs = []
    for pair_text in pairs:
        # Bây giờ hàm có thể truy cập `prefix` và `suffix` một cách chính xác
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

def rerank_documents(reranker_data, query, documents, batch_size=4):
    model = reranker_data["model"]
    tokenizer = reranker_data["tokenizer"]
    task_instruction = 'Given a user query about computer components or pre-built PCs, determine if the following document is relevant.'
    
    rerank_pairs = [format_instruction(task_instruction, query, doc) for doc in documents]
    
    all_scores = []
    for i in tqdm(range(0, len(rerank_pairs), batch_size), desc="Reranking Batches"):
        batch_pairs = rerank_pairs[i:i + batch_size]
        
        # Lệnh gọi này giờ đã khớp với định nghĩa hàm đã sửa
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
        
        # SỬA LỖI 2: Thay `torch_dtype` bằng `dtype` để hết cảnh báo
        reranker_model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_NAME, 
            dtype=torch.float16 # Sửa ở đây
        ).to(device).eval()
        
        print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load models. Error: {e}")
        return
        
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
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

    print(f"\nConnecting to ChromaDB at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    prebuilt_collection = chroma_client.get_collection(name="prebuilt_pcs")
    print("Successfully connected to ChromaDB.")

    print(f"\nGenerating embedding for query: '{SAMPLE_QUERY}'")
    query_embedding = embedding_model.encode(SAMPLE_QUERY, prompt_name="query")
    
    num_candidates = 20
    print(f"Retrieving top {num_candidates} candidates from prebuilt collection...")
    results_prebuilt = prebuilt_collection.query(query_embeddings=[query_embedding.tolist()], n_results=num_candidates)
    initial_documents = results_prebuilt['documents'][0]
    initial_metadatas = results_prebuilt['metadatas'][0]
    print(f"Found {len(initial_documents)} total candidates for reranking.")

    scores = rerank_documents(reranker_data, SAMPLE_QUERY, initial_documents)

    results_with_scores = sorted(list(zip(scores, initial_metadatas)), key=lambda x: x[0], reverse=True)
    
    top_n_reranked = 5
    print(f"\n--- TOP {top_n_reranked} RERANKED RESULTS ---")
    for i, (score, meta) in enumerate(results_with_scores[:top_n_reranked]):
        print(f"\n{i+1}. Score: {score:.4f}")
        print(f"   Type: Pre-built PC\n   Use Case: {meta.get('Nhu cầu', 'N/A')}\n   Price: {meta.get('PRICE', 'N/A')}")

if __name__ == "__main__":
    main()