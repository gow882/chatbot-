import chromadb
import csv
import sys
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "./data"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chromadb")
PREBUILT_CSV_PATH = os.path.join(DATA_DIR, "prebuilt-pc.csv")
WN_CSV_PATH = os.path.join(DATA_DIR, "wn.csv")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_ROWS_TO_PROCESS = 1000  # Giới hạn số lượng truyện nạp vào để chạy test nhanh trên CPU

# --- LOAD EMBEDDING MODEL ---
def load_model():
    """Tải mô hình embedding SentenceTransformer."""
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    try:
        # Tự động chọn thiết bị (CUDA nếu có, nếu không thì CPU)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Could not load the model. Error: {e}")
        sys.exit(1)

def create_embeddings_from_csv(file_path, text_columns, metadata_columns, output_dir='data'):
    """
    Reads data from a CSV file, creates embeddings, and saves them.

    Args:
        file_path (str): Path to the input CSV file.
        text_columns (list): List of columns to combine for creating the text to be embedded.
        metadata_columns (list): List of all columns to be saved as metadata.
        output_dir (str): Directory to save the output files.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path)
    df = df.dropna(subset=text_columns)
    df['combined_text'] = df[text_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)

    documents = df['combined_text'].tolist()
    metadata = df[metadata_columns].to_dict(orient='records')

    return documents, metadata

def main():
    """Hàm chính để chạy quy trình embedding và ingestion."""
    model = load_model()

    # --- SETUP CHROMA DB ---
    if not os.path.exists(PREBUILT_CSV_PATH) and not os.path.exists(WN_CSV_PATH):
        print(f"Lỗi: Không tìm thấy file CSV nào trong thư mục '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Initializing ChromaDB client at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    prebuilt_collection = chroma_client.get_or_create_collection(name="prebuilt_pcs")
    wn_collection = chroma_client.get_or_create_collection(name="web_novels")

    # --- PROCESS PRE-BUILT PCs (prebuilt-pc.csv) ---
    prebuilt_documents, prebuilt_ids, prebuilt_metadatas = [], [], []
    if os.path.exists(PREBUILT_CSV_PATH):
        print("\nProcessing pre-built PCs from prebuilt-pc.csv...")
        with open(PREBUILT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(tqdm(list(reader), desc="Reading pre-built PCs")):
                doc_parts = [
                    f"Nhu cầu: {row.get('Nhu cầu')}",
                    f"CPU: {row.get('Hãng CPU')} {row.get('CPU')}",
                    f"MAIN: {row.get('MAIN')}",
                    f"RAM: {row.get('RAM')}",
                    f"VGA: {row.get('VGA')}",
                    f"Storage: {row.get('Storage')}",
                    f"PRICE: {row.get('PRICE')}"
                ]
                doc = ". ".join(filter(None, doc_parts))
                prebuilt_documents.append(doc)
                prebuilt_ids.append(f"prebuilt_{i+1}")
                meta = dict(row)
                link_value = meta.get('LINK') or meta.get('LINK SP') or ''
                meta['LINK'] = link_value
                prebuilt_metadatas.append(meta)

        print(f"Generating embeddings for {len(prebuilt_documents)} pre-built PCs...")
        prebuilt_embeddings = model.encode(prebuilt_documents, show_progress_bar=True)
        prebuilt_collection.upsert(embeddings=prebuilt_embeddings.tolist(), documents=prebuilt_documents, ids=prebuilt_ids, metadatas=prebuilt_metadatas)
        print(f"Successfully loaded {prebuilt_collection.count()} pre-built PCs into ChromaDB.")
    else:
        print(f"Skipping pre-built PCs: {PREBUILT_CSV_PATH} not found.")

    # --- PROCESS WEB NOVELS (wn.csv) ---
    if os.path.exists(WN_CSV_PATH):
        print(f"\nProcessing Web Novels from {WN_CSV_PATH}...")
        df_wn = pd.read_csv(WN_CSV_PATH)
        
        # --- TỐI ƯU TỐC ĐỘ: Giới hạn số lượng truyện nếu quá lớn ---
        if len(df_wn) > MAX_ROWS_TO_PROCESS:
            print(f"Dataset quá lớn ({len(df_wn)} truyện). Chỉ cắt {MAX_ROWS_TO_PROCESS} truyện đầu tiên để nạp nhanh.")
            df_wn = df_wn.head(MAX_ROWS_TO_PROCESS)
            
        # Using a subset of columns for embedding to keep it clean and relevant
        text_cols = ['title', 'genres', 'tags', 'description']
        df_wn['combined_text'] = df_wn[text_cols].fillna('').apply(lambda row: ' | '.join(row.astype(str)), axis=1)
        
        wn_documents = df_wn['combined_text'].tolist()
        wn_ids = [f"wn_{row['novel_id']}" if 'novel_id' in row and pd.notna(row['novel_id']) else f"wn_index_{i}" for i, row in df_wn.iterrows()]
        
        # Select important metadata columns
        meta_cols = ['novel_id', 'url', 'title', 'rating', 'chapters', 'authors', 'language']
        # Remove NaNs in metadata as ChromaDB doesn't accept nulls in metadata
        df_wn_meta = df_wn[meta_cols].copy()
        df_wn_meta = df_wn_meta.fillna('')
        # Convert any remaining non-string/int/float types to string
        for col in df_wn_meta.columns:
            df_wn_meta[col] = df_wn_meta[col].astype(str)
            
        wn_metadatas = df_wn_meta.to_dict(orient='records')
        
        print(f"Generating embeddings and uploading {len(wn_documents)} Web Novels in batches...")
        batch_size = 500  # Process in batches to prevent OOM
        for i in tqdm(range(0, len(wn_documents), batch_size), desc="Web Novel Batches"):
            batch_docs = wn_documents[i:i+batch_size]
            batch_ids = wn_ids[i:i+batch_size]
            batch_metas = wn_metadatas[i:i+batch_size]
            
            # Embed batch
            batch_embeds = model.encode(batch_docs, show_progress_bar=False)
            wn_collection.upsert(
                embeddings=batch_embeds.tolist(),
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas
            )
        print(f"Successfully loaded Web Novels. Total in Chroma queryable: {wn_collection.count()}")
    else:
        print(f"Skipping Web Novels: {WN_CSV_PATH} not found.")
    
    print("\n✅ Embedding and ingestion complete!")

if __name__ == "__main__":
    main()