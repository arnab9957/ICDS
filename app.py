import json
import os
import csv
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from datetime import datetime

# Page configuration for a premium look
st.set_page_config(
    page_title="Project ICDS - Sandbox Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for rich aesthetics (Dark theme compatibility)
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 15px;
    }
    .candidate-card {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #334155;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: #FF8F60;
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define static constants (from rank.py)
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "l&t", 
    "ltts", "ltimindtree", "deloitte", "pwc", "ey", "kpmg", "mphasis", 
    "hexaware", "ust global", "genpact", "cognizant technology solutions"
}

NLP_IR_KEYWORDS = {
    "nlp", "retrieval", "search", "embedding", "transformer", "bert", "gpt", 
    "llm", "language", "text", "semantic", "recommend", "ranking", "rag", 
    "information retrieval", "vector search", "sentence-transformers", "haystack", 
    "llama", "qdrant", "milvus", "weaviate", "pinecone", "faiss"
}

CV_SPEECH_KEYWORDS = {
    "cv", "vision", "image", "object detection", "speech", "audio", "voice", 
    "tts", "asr", "robotics", "robot", "opencv", "speech recognition", "yolo", 
    "gans", "image classification"
}

RESEARCH_KEYWORDS = {"researcher", "research scientist", "academic", "professor", "phd student", "postdoc", "lab"}
PRODUCTION_KEYWORDS = {"production", "deploy", "scale", "ship", "engineer", "infrastructure", "system", "pipeline", "product"}

# Cached resources for ONNX Session
@st.cache_resource
def load_onnx_model():
    model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx")
    tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options)
    
    return tokenizer, session

# Embed helper
def get_embeddings(texts, tokenizer, session):
    input_names = [inp.name for inp in session.get_inputs()]
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        
    outputs = session.run(None, inputs)
    token_embeddings = outputs[0]
    
    # Mean Pooling
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    batch_embs = sum_embeddings / sum_mask
    
    # L2 Normalize
    norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
    return batch_embs / np.clip(norms, a_min=1e-9, a_max=None)

# Honeypot rules
def is_honeypot(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    
    yoe = profile.get("years_of_experience", 0)
    total_yoe_months = yoe * 12.0
    
    # Rule 1: Single role duration claimed exceeds total profile experience
    for r in history:
        dur = r.get("duration_months", 0)
        if dur > total_yoe_months + 0.5:
            return True, f"Role duration ({dur} mos) exceeds total experience ({total_yoe_months:.1f} mos)"
            
    # Rule 2: Claimed role duration differs from start/end dates
    for r in history:
        start_str = r.get("start_date")
        end_str = r.get("end_date") or "2026-06-24"
        duration_months = r.get("duration_months", 0)
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d")
            actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if abs(actual_months - duration_months) > 6:
                return True, f"Claimed duration {duration_months} mos, but date interval is {actual_months} mos"
        except Exception:
            pass
            
    # Rule 3: Extreme discrepancy between profile YoE and sum of all history durations
    total_history_months = sum(r.get("duration_months", 0) for r in history)
    if yoe > 5.0 and total_history_months < 24:
        return True, f"Claimed {yoe} YoE, but career history sum is only {total_history_months} mos"
    elif yoe < 3.0 and total_history_months > 96:
        return True, f"Claimed {yoe} YoE, but career history sum is {total_history_months} mos"
        
    # Rule 4: Expert proficiency but 0 usage
    expert_zero_dur = [s for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
    if len(expert_zero_dur) >= 3:
        return True, f"Expert/Advanced proficiency with 0 duration in {len(expert_zero_dur)} skills"
        
    return False, ""

# Extractor helpers (from rank.py)
def extract_vector_db_skill(skills, headline, summary):
    vector_dbs = ["pinecone", "weaviate", "milvus", "qdrant", "faiss", "elasticsearch", "opensearch", "pgvector"]
    for s in skills:
        name = s.get("name", "").strip()
        if name.lower() in vector_dbs:
            return name
    for s in skills:
        name = s.get("name", "").strip()
        for db in vector_dbs:
            if db in name.lower():
                return name
    combined = f"{headline} {summary}".lower()
    for db in vector_dbs:
        if db in combined:
            return db.capitalize() if db != "pgvector" else "pgvector"
    ml_skills = ["nlp", "retrieval", "search", "embeddings", "transformers", "llm", "machine learning"]
    for s in skills:
        name = s.get("name", "").strip()
        for ms in ml_skills:
            if ms in name.lower():
                return name
    if skills:
        return skills[0].get("name", "")
    return "applied machine learning"

def extract_eval_skill(skills, headline, summary):
    eval_frameworks = ["ndcg", "mrr", "map", "learning to rank", "ltr", "evaluation"]
    for s in skills:
        name = s.get("name", "").strip()
        if name.lower() in eval_frameworks:
            return name
    for s in skills:
        name = s.get("name", "").strip()
        for ef in eval_frameworks:
            if ef in name.lower():
                return name
    combined = f"{headline} {summary}".lower()
    for ef in eval_frameworks:
        if ef in combined:
            return ef.upper() if len(ef) <= 4 else ef.title()
    eng_skills = ["python", "sql", "git", "backend", "docker", "aws"]
    for s in skills:
        name = s.get("name", "").strip()
        for es in eng_skills:
            if es in name.lower():
                return name
    if len(skills) > 1:
        return skills[1].get("name", "")
    elif skills:
        return skills[0].get("name", "")
    return "evaluation frameworks"

# Streamlit App Logic
st.markdown('<div class="main-title">Project ICDS 🎯</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Candidate Discovery & Sandbox Ranker — Redrob Hackathon v4</div>', unsafe_allow_html=True)

# Loading model
with st.spinner("Initializing ONNX embedding model (Xenova/all-MiniLM-L6-v2)..."):
    tokenizer, session = load_onnx_model()

# Sidebar Setup
st.sidebar.title("Sandbox Configuration")
st.sidebar.markdown("This environment matches the **exact local CPU constraints** of the validation test harness.")

# Precompute query embedding
jd_query = st.sidebar.text_area(
    "Job Description Query (Intent)", 
    value="Vector Databases (Pinecone, Weaviate, Milvus, etc.), evaluation frameworks (NDCG, MRR, MAP), and Python proficiency.",
    height=100
)

# Uploading data
st.sidebar.subheader("Ingestion Layer")
uploaded_file = st.sidebar.file_uploader("Upload candidates.jsonl (≤100 recommended for sandbox)", type=["jsonl", "json"])

# Create sample candidate data as fallback
@st.cache_data
def get_sample_candidates():
    # A few representative sample candidates (including one honeypot and one match)
    return [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "Senior AI Engineer",
                "headline": "ML Engineer | Building semantic search and vector databases",
                "summary": "5+ years of experience shipping production RAG systems. Proficient in Pinecone and evaluation metrics like NDCG.",
                "years_of_experience": 6.5,
                "location": "Pune, India",
                "country": "India"
            },
            "skills": [
                {"name": "Python", "duration_months": 72, "proficiency": "expert"},
                {"name": "Pinecone", "duration_months": 24, "proficiency": "expert"},
                {"name": "NDCG", "duration_months": 36, "proficiency": "advanced"},
                {"name": "NLP", "duration_months": 48, "proficiency": "advanced"}
            ],
            "career_history": [
                {"company": "HyperML", "title": "Senior Machine Learning Engineer", "duration_months": 36, "is_current": True, "start_date": "2023-06-24", "description": "Designed and shipped vector databases using Pinecone. Implemented offline NDCG evaluations."},
                {"company": "AppTech Solutions", "title": "ML Engineer", "duration_months": 42, "is_current": False, "start_date": "2019-12-01", "end_date": "2023-06-01", "description": "Wrote code for NLP classification pipelines."}
            ],
            "redrob_signals": {
                "notice_period_days": 15,
                "recruiter_response_rate": 0.85,
                "last_active_date": "2026-06-20",
                "willing_to_relocate": True,
                "open_to_work_flag": True
            }
        },
        {
            "candidate_id": "CAND_0000002",
            "profile": {
                "current_title": "Marketing Manager",
                "headline": "Expert in LLMs, RAG, LangChain, Pinecone, Milvus, Weaviate",
                "summary": "10 years experience doing advertising, but also expert in everything AI.",
                "years_of_experience": 10.0,
                "location": "Dallas, USA",
                "country": "USA"
            },
            "skills": [
                {"name": "LangChain", "duration_months": 3, "proficiency": "expert"},
                {"name": "OpenAI", "duration_months": 3, "proficiency": "expert"}
            ],
            "career_history": [
                {"company": "Marketing Corp", "title": "Marketing Manager", "duration_months": 120, "is_current": True, "start_date": "2016-06-24", "description": "Managing ad accounts and newsletters."}
            ],
            "redrob_signals": {
                "notice_period_days": 90,
                "recruiter_response_rate": 0.05,
                "last_active_date": "2025-10-10",
                "willing_to_relocate": False,
                "open_to_work_flag": False
            }
        },
        {
            "candidate_id": "CAND_0000003",
            "profile": {
                "current_title": "Fake Expert Candidate",
                "headline": "AI Guru",
                "summary": "Unbelievable skills.",
                "years_of_experience": 2.0,
                "location": "Mumbai, India",
                "country": "India"
            },
            "skills": [
                {"name": "Pinecone", "duration_months": 0, "proficiency": "expert"},
                {"name": "Weaviate", "duration_months": 0, "proficiency": "expert"},
                {"name": "Milvus", "duration_months": 0, "proficiency": "expert"}
            ],
            "career_history": [
                {"company": "Startup", "title": "Engineer", "duration_months": 120, "is_current": False, "start_date": "2024-01-01", "end_date": "2024-06-01"}
            ],
            "redrob_signals": {
                "notice_period_days": 30,
                "recruiter_response_rate": 0.90,
                "last_active_date": "2026-06-24",
                "willing_to_relocate": True
            }
        }
    ]

# Select raw data source
candidates = []
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".jsonl"):
            for line in uploaded_file:
                if line.strip():
                    candidates.append(json.loads(line))
        else:
            candidates = json.load(uploaded_file)
        st.sidebar.success(f"Successfully loaded {len(candidates)} candidates.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
else:
    candidates = get_sample_candidates()
    st.sidebar.info("Using built-in sample candidates. Upload a custom JSONL to test your own data.")

# Run ranking when requested
if st.button("🚀 Execute Ranking Pipeline"):
    if not candidates:
        st.warning("No candidates loaded.")
    else:
        # Precompute Query Vector
        query_vector = get_embeddings([jd_query], tokenizer, session)[0]
        
        # Scoring variables
        ranks = []
        filtered_count = 0
        honeypot_count = 0
        
        # Prepare candidate descriptions to encode
        candidates_to_encode = []
        cands_meta = []
        
        for cand in candidates:
            # Check L0 Honeypot Filter
            is_hp, hp_reason = is_honeypot(cand)
            if is_hp:
                honeypot_count += 1
                continue
                
            profile = cand.get("profile", {})
            skills = cand.get("skills", [])
            history = cand.get("career_history", [])
            
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            
            # Check L0 Filters: CV/Speech vs NLP/IR
            history_texts = []
            for r in history:
                history_texts.append(r.get("title", ""))
                history_texts.append(r.get("description", ""))
            history_combined = " ".join(history_texts)
            
            combined_text = f"{headline} {summary} " + " ".join([s.get("name", "") for s in skills]) + " " + history_combined
            combined_lower = combined_text.lower()
            
            has_nlp_ir = any(kw in combined_lower for kw in NLP_IR_KEYWORDS)
            has_cv_speech = any(kw in combined_lower for kw in CV_SPEECH_KEYWORDS)
            
            if has_cv_speech and not has_nlp_ir:
                filtered_count += 1
                continue
                
            # L0: Research vs Production
            has_research = any(kw in combined_lower for kw in RESEARCH_KEYWORDS)
            has_production = any(kw in combined_lower for kw in PRODUCTION_KEYWORDS)
            if has_research and not has_production:
                filtered_count += 1
                continue
                
            # Ready for embedding
            title = profile.get("current_title", "")
            top_skills = [s.get("name", "") for s in skills[:10]]
            repr_text = f"{title} | {headline} | {', '.join(top_skills)}"
            
            candidates_to_encode.append(repr_text)
            cands_meta.append(cand)
            
        if not candidates_to_encode:
            st.error("All candidates filtered out at Layer 0 (Honeypots, academic researchers, or CV/Speech-only engineers).")
        else:
            # Batch encode candidates
            with st.spinner("Computing semantic embeddings for candidate profiles..."):
                cand_vectors = get_embeddings(candidates_to_encode, tokenizer, session)
                
            # Cosine similarities
            similarities = np.dot(cand_vectors, query_vector)
            
            # Apply heuristics
            scored_candidates = []
            for idx, cand in enumerate(cands_meta):
                profile = cand.get("profile", {})
                skills = cand.get("skills", [])
                history = cand.get("career_history", [])
                signals = cand.get("redrob_signals", {})
                
                cos_sim = float(similarities[idx])
                
                # Tenure calculation
                company_durations = {}
                for r in history:
                    comp = r.get("company", "").strip().lower()
                    if comp:
                        company_durations[comp] = company_durations.get(comp, 0) + r.get("duration_months", 0)
                
                num_companies = len(company_durations)
                total_dur = sum(company_durations.values())
                avg_tenure = total_dur / num_companies if num_companies > 0 else 0.0
                
                tenure_penalty = 1.0
                if num_companies > 0 and avg_tenure < 18.0:
                    tenure_penalty = 0.5
                    
                # Consulting firm penalty
                all_consulting = True
                has_product = False
                for r in history:
                    comp = r.get("company", "").strip().lower()
                    is_consulting = False
                    for c in CONSULTING_COMPANIES:
                        if c in comp:
                            is_consulting = True
                            break
                    if not is_consulting:
                        has_product = True
                        all_consulting = False
                        
                if not history:
                    all_consulting = False
                    
                company_multiplier = 1.0
                if all_consulting:
                    company_multiplier = 0.1
                elif not has_product:
                    company_multiplier = 0.5
                    
                fit_score = cos_sim * tenure_penalty * company_multiplier
                
                # LangChain / OpenAI short-term experience penalty
                ai_skills = [s for s in skills if any(kw in s.get("name", "").lower() for kw in ["langchain", "openai", "gpt-4", "llm", "rag"])]
                pre_llm_ml_skills = [s for s in skills if any(kw in s.get("name", "").lower() for kw in ["scikit-learn", "tensorflow", "pytorch", "nlp", "information retrieval", "vector search", "regression", "classification", "svm", "random forest", "xgboost", "clustering", "spacy", "nltk"])]
                if ai_skills and not pre_llm_ml_skills:
                    if all(s.get("duration_months", 0) <= 12 for s in ai_skills):
                        fit_score *= 0.1
                        
                # Non-coding management role penalty
                recent_roles_are_non_coding = True
                has_recent_role = False
                for r in history:
                    is_current = r.get("is_current", False)
                    end_str = r.get("end_date")
                    is_recent = False
                    if is_current:
                        is_recent = True
                    elif end_str:
                        try:
                            end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                            if (datetime(2026, 6, 24) - end_dt).days <= 540:
                                is_recent = True
                        except Exception:
                            pass
                    if is_recent:
                        has_recent_role = True
                        r_title = r.get("title", "").lower()
                        r_desc = r.get("description", "").lower()
                        coding_keywords = ["engineer", "developer", "programmer", "coder", "architect", "scientist", "build", "write", "develop", "implement", "code", "ship"]
                        management_titles = ["manager", "director", "vp", "lead", "head"]
                        is_management = any(kw in r_title for kw in management_titles)
                        is_coding_title = any(kw in r_title for kw in coding_keywords)
                        if is_coding_title or (not is_management and any(kw in r_desc for kw in ["code", "develop", "implement", "write", "ship"])):
                            recent_roles_are_non_coding = False
                            break
                if has_recent_role and recent_roles_are_non_coding:
                    fit_score *= 0.3
                    
                # Layer 2: Behavioral Multipliers
                notice = signals.get("notice_period_days", 90)
                notice_mult = 1.15 if notice <= 30 else 1.0
                
                resp_rate = signals.get("recruiter_response_rate", 0.0)
                resp_mult = 1.0 if resp_rate >= 0.3 else 0.5
                
                last_active_str = signals.get("last_active_date", "")
                active_mult = 1.0
                days_inactive = 0
                if last_active_str:
                    try:
                        la_dt = datetime.strptime(last_active_str, "%Y-%m-%d")
                        curr_dt = datetime(2026, 6, 24)
                        days_inactive = (curr_dt - la_dt).days
                        if days_inactive > 180:
                            active_mult = 0.3
                    except Exception:
                        pass
                        
                calibrated_score = fit_score * notice_mult * resp_mult * active_mult
                
                # Metadata
                willing_relocate = signals.get("willing_to_relocate", False)
                loc = profile.get("location", "")
                yoe = profile.get("years_of_experience", 0.0)
                cid = cand["candidate_id"]
                
                scored_candidates.append({
                    "candidate_id": cid,
                    "score": calibrated_score,
                    "yoe": yoe,
                    "skills": skills,
                    "headline": headline,
                    "summary": summary,
                    "notice": notice,
                    "avg_tenure": avg_tenure,
                    "resp_rate": resp_rate,
                    "days_inactive": days_inactive,
                    "willing_relocate": willing_relocate,
                    "loc": loc
                })
                
            # Layer 3: Sorting & Tie-Breaking
            scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
            
            # Generate justifications
            output_rows = []
            for rank_idx, cand_info in enumerate(scored_candidates):
                cid = cand_info["candidate_id"]
                score = cand_info["score"]
                yoe = cand_info["yoe"]
                skills = cand_info["skills"]
                headline = cand_info["headline"]
                summary = cand_info["summary"]
                notice = cand_info["notice"]
                avg_tenure = cand_info["avg_tenure"]
                resp_rate = cand_info["resp_rate"]
                days_inactive = cand_info["days_inactive"]
                willing_relocate = cand_info["willing_relocate"]
                loc = cand_info["loc"]
                
                top_vector_db = extract_vector_db_skill(skills, headline, summary)
                top_eval_skill = extract_eval_skill(skills, headline, summary)
                
                concerns = []
                if notice > 30:
                    concerns.append(f"notice period of {notice} days")
                if avg_tenure > 0 and avg_tenure < 18.0:
                    concerns.append("slightly jumpy tenure history")
                if days_inactive > 180:
                    concerns.append("inactive for over 6 months")
                if resp_rate < 0.3:
                    concerns.append("low recruiter response rate")
                if not willing_relocate and all(c not in loc.lower() for c in ["pune", "noida"]):
                    concerns.append("unwilling to relocate to Pune/Noida")
                    
                concern_str = ""
                if concerns:
                    concern_str = "Concern: " + " and ".join(concerns) + "."
                    
                if rank_idx < 15:
                    prefix = "Outstanding candidate: "
                elif rank_idx < 50:
                    prefix = "Strong candidate: "
                else:
                    prefix = "Decent adjacent fit: "
                    
                structures = [
                    lambda prefix, yoe, top_vdb, top_eval, concern: f"{prefix}{yoe:.1f} years of applied ML experience. Matches JD via {top_vdb} and {top_eval}. {concern}".strip(),
                    lambda prefix, yoe, top_vdb, top_eval, concern: f"{prefix}Brings {yoe:.1f} years of ML experience. Good fit with skills in {top_vdb} and {top_eval}. {concern}".strip(),
                    lambda prefix, yoe, top_vdb, top_eval, concern: f"{prefix}Matches JD with {yoe:.1f} years in applied ML, showcasing strong command over {top_vdb} & {top_eval}. {concern}".strip()
                ]
                
                template_fn = structures[rank_idx % len(structures)]
                reasoning = template_fn(prefix, yoe, top_vector_db, top_eval_skill, concern_str)
                
                output_rows.append({
                    "candidate_id": cid,
                    "rank": rank_idx + 1,
                    "score": f"{score:.4f}",
                    "reasoning": reasoning
                })
                
            # Create metrics cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><h4>Ingested Candidates</h4><h3>{len(candidates)}</h3></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><h4>Honeypots Removed</h4><h3>{honeypot_count}</h3></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><h4>Academic/CV Filtered</h4><h3>{filtered_count}</h3></div>', unsafe_allow_html=True)
                
            # Present Results in a Dataframe
            df_out = pd.DataFrame(output_rows)
            st.subheader("Ranked Candidates (Top matches first)")
            st.dataframe(df_out, use_container_width=True)
            
            # Expose download link
            csv_data = df_out.to_csv(index=False)
            st.download_button(
                label="📥 Download Submission CSV",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )
            
            # Details view for candidates
            st.subheader("Search & Deep Dive Candidate Record")
            selected_id = st.selectbox("Select Candidate ID to inspect parsed details:", df_out["candidate_id"].tolist())
            
            selected_cand = next((c for c in scored_candidates if c["candidate_id"] == selected_id), None)
            if selected_cand:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Headline:**", selected_cand["headline"])
                    st.write("**Years of Experience:**", selected_cand["yoe"])
                    st.write("**Location:**", selected_cand["loc"])
                    st.write("**Stated Notice Period:**", f"{selected_cand['notice']} days")
                with c2:
                    st.write("**Recruiter Response Rate:**", f"{selected_cand['resp_rate']:.0%}")
                    st.write("**Days Inactive:**", f"{selected_cand['days_inactive']} days")
                    st.write("**Average Company Tenure:**", f"{selected_cand['avg_tenure']:.1f} months")
                    st.write("**Willing to Relocate:**", "Yes" if selected_cand["willing_relocate"] else "No")
                    
                st.write("**Skills:**")
                skills_md = ""
                for s in selected_cand["skills"]:
                    skills_md += f"<span class='tag-badge' style='background-color:#3b82f6;color:white;'>{s.get('name')} ({s.get('proficiency')})</span>"
                st.markdown(skills_md, unsafe_allow_html=True)
