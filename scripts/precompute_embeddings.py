import json
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import time

def precompute_embeddings(candidates_path, embeddings_out, ids_out, batch_size=512):
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
    
    candidate_ids = []
    texts_to_encode = []
    
    print(f"Reading candidates from {candidates_path}...")
    start_time = time.time()
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            skills = cand.get("skills", [])
            
            title = profile.get("current_title", "")
            headline = profile.get("headline", "")
            top_skills = [s.get("name", "") for s in skills[:10]]
            
            # Combine into a concise representation
            text = f"{title} | {headline} | {', '.join(top_skills)}"
            
            candidate_ids.append(cid)
            texts_to_encode.append(text)
            
    total_candidates = len(candidate_ids)
    print(f"Loaded {total_candidates} candidates. Starting encoding...")
    
    # Encode in batches
    embeddings = []
    num_batches = (total_candidates + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_candidates)
        
        batch_texts = texts_to_encode[start_idx:end_idx]
        
        # Tokenize
        encodings = tokenizer.encode_batch(batch_texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "token_type_ids" in input_names:
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            inputs["token_type_ids"] = token_type_ids
            
        # ONNX inference
        outputs = session.run(None, inputs)
        token_embeddings = outputs[0]
        
        # Mean Pooling
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        batch_embs = sum_embeddings / sum_mask
        
        # L2 Normalize
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        batch_embs = batch_embs / np.clip(norms, a_min=1e-9, a_max=None)
        
        embeddings.append(batch_embs)
        
        if (i + 1) % 20 == 0 or i == num_batches - 1:
            elapsed = time.time() - start_time
            avg_rate = (end_idx) / elapsed
            est_total = total_candidates / avg_rate
            remaining = est_total - elapsed
            print(f"Processed batch {i+1}/{num_batches} (candidates {end_idx}/{total_candidates}). "
                  f"Rate: {avg_rate:.1f} cands/sec. Est. remaining: {remaining:.1f} sec.")
            
    all_embeddings = np.vstack(embeddings)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(embeddings_out), exist_ok=True)
    
    # Save files
    print(f"Saving embeddings matrix of shape {all_embeddings.shape} to {embeddings_out}...")
    np.save(embeddings_out, all_embeddings)
    
    print(f"Saving candidate ID mapping to {ids_out}...")
    with open(ids_out, "w", encoding="utf-8") as f_ids:
        json.dump(candidate_ids, f_ids)
        
    print(f"Pre-computation successfully completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    precompute_embeddings(
        candidates_path="data/candidates.jsonl",
        embeddings_out="data/candidate_embeddings.npy",
        ids_out="data/candidate_ids.json"
    )
