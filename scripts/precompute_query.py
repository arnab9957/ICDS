import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os

def precompute_query(query_text, output_path):
    print("Downloading model and tokenizer from HuggingFace...")
    model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx")
    tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    
    print("Loading ONNX Inference Session...")
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options)
    
    input_names = [inp.name for inp in session.get_inputs()]
    
    print(f"Encoding query: '{query_text}'...")
    encodings = tokenizer.encode_batch([query_text])
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    if "token_type_ids" in input_names:
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        inputs["token_type_ids"] = token_type_ids
        
    outputs = session.run(None, inputs)
    token_embeddings = outputs[0]
    
    # Mean pooling
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    query_emb = sum_embeddings / sum_mask
    
    # L2 Normalize
    norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.clip(norms, a_min=1e-9, a_max=None)
    query_emb_np = query_emb[0]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, query_emb_np)
    print(f"Saved query embedding of shape {query_emb_np.shape} to {output_path}")

if __name__ == "__main__":
    precompute_query(
        query_text="Vector Databases (Pinecone, Weaviate, Milvus, etc.), evaluation frameworks (NDCG, MRR, MAP), and Python proficiency.",
        output_path="data/query_embedding.npy"
    )
