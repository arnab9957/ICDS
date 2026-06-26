import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import time

def test_onnx():
    try:
        print("Downloading model and tokenizer from HuggingFace...")
        model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")
        print(f"Model path: {model_path}")
        print(f"Tokenizer path: {tokenizer_path}")
        
        print("Loading tokenizer...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        print("Loading ONNX Inference Session...")
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options)
        
        sentences = ["Hello world", "Machine learning and NLP vector databases"]
        
        print("Tokenizing...")
        encodings = tokenizer.encode_batch(sentences)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        input_names = [inp.name for inp in session.get_inputs()]
        print("Model inputs expected:", input_names)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "token_type_ids" in input_names:
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            inputs["token_type_ids"] = token_type_ids
            
        print("Running inference...")
        start = time.time()
        outputs = session.run(None, inputs)
        token_embeddings = outputs[0]
        print(f"Inference complete in {time.time() - start:.4f}s. Output shape: {token_embeddings.shape}")
        
        # Mean Pooling - fix method here
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        print("Pooled embeddings shape:", embeddings.shape)
        print("Cosine similarity between the two:", np.dot(embeddings[0], embeddings[1]))
        print("SUCCESS!")
    except Exception as e:
        print("FAILED with error:", str(e))

if __name__ == "__main__":
    test_onnx()
