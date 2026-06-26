import json
import os
import csv
import argparse
from datetime import datetime
import numpy as np

# Consulting companies list
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "l&t", 
    "ltts", "ltimindtree", "deloitte", "pwc", "ey", "kpmg", "mphasis", 
    "hexaware", "ust global", "genpact", "cognizant technology solutions"
}

# L0 Filtering Keywords
NLP_IR_KEYWORDS = {
    "nlp", "retrieval", "search", "embedding", "transformer", "bert", "gpt", 
    "llm", "language", "text", "semantic", "recommend", "ranking", "rag", 
    "information retrieval", "vector search", "sentence-transformers", "haystack", 
    "llama", "qdrant", "milvus", "weaviate", "pinecone", "faiss"
}

CV_SPEECH_KEYWORDS = {
    "cv", "vision", "image", "object detection", "speech", "audio", "voice", 
    "tts", "asr", "robotics", "robot", "opencv", "speech recognition", "yolo", 
    "gans", "gans", "image classification"
}

RESEARCH_KEYWORDS = {"researcher", "research scientist", "academic", "professor", "phd student", "postdoc", "lab"}
PRODUCTION_KEYWORDS = {"production", "deploy", "scale", "ship", "engineer", "infrastructure", "system", "pipeline", "product"}

def load_precomputed_data():
    candidate_ids = []
    embeddings = None
    query_emb = None
    honeypots = {}
    
    # Load candidate IDs
    ids_path = "data/candidate_ids.json"
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            candidate_ids = json.load(f)
            
    # Load embeddings matrix
    embs_path = "data/candidate_embeddings.npy"
    if os.path.exists(embs_path):
        embeddings = np.load(embs_path)
        
    # Load query embedding
    query_path = "data/query_embedding.npy"
    if os.path.exists(query_path):
        query_emb = np.load(query_path)
        
    # Load honeypots
    hp_path = "data/honeypots.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r", encoding="utf-8") as f:
            honeypots = json.load(f)
            
    return candidate_ids, embeddings, query_emb, honeypots

def get_fallback_similarity(cand):
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    
    text = f"{profile.get('current_title', '')} {profile.get('headline', '')} {profile.get('summary', '')} " + \
           " ".join([s.get("name", "") for s in skills])
    text_lower = text.lower()
    
    # Core requirements match count
    score = 0.0
    core_keywords = ["vector", "pinecone", "weaviate", "milvus", "qdrant", "faiss", "ndcg", "mrr", "map", "python", "nlp", "retrieval", "search", "llm", "rag"]
    for kw in core_keywords:
        if kw in text_lower:
            score += 1.0
            
    return min(score / 10.0, 1.0)

def extract_vector_db_skill(skills, headline, summary):
    vector_dbs = ["pinecone", "weaviate", "milvus", "qdrant", "faiss", "elasticsearch", "opensearch", "pgvector"]
    
    # Try skills first
    for s in skills:
        name = s.get("name", "").strip()
        if name.lower() in vector_dbs:
            return name
    for s in skills:
        name = s.get("name", "").strip()
        for db in vector_dbs:
            if db in name.lower():
                return name
                
    # Search headline/summary
    combined = f"{headline} {summary}".lower()
    for db in vector_dbs:
        if db in combined:
            return db.capitalize() if db != "pgvector" else "pgvector"
            
    # Fallback to other relevant ML/AI skills they actually have to avoid hallucination
    ml_skills = ["nlp", "retrieval", "search", "embeddings", "transformers", "llm", "machine learning", "deep learning", "python", "pytorch", "tensorflow"]
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
    
    # Try skills first
    for s in skills:
        name = s.get("name", "").strip()
        if name.lower() in eval_frameworks:
            return name
    for s in skills:
        name = s.get("name", "").strip()
        for ef in eval_frameworks:
            if ef in name.lower():
                return name
                
    # Search headline/summary
    combined = f"{headline} {summary}".lower()
    for ef in eval_frameworks:
        if ef in combined:
            return ef.upper() if len(ef) <= 4 else ef.title()
            
    # Fallback to other technical skills they actually have to avoid hallucination
    eng_skills = ["python", "sql", "git", "backend", "docker", "aws", "data engineering", "spark", "airflow"]
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

def parse_args():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob hackathon.")
    parser.add_argument("--candidates", required=True, help="Path to input candidates.jsonl file")
    parser.add_argument("--out", required=True, help="Path to write output submission.csv file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load precomputed data
    candidate_ids, embeddings, query_emb, honeypots = load_precomputed_data()
    cand_id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}
    
    # Pre-calculate similarities if embeddings are loaded
    similarities = None
    if embeddings is not None and query_emb is not None:
        dot_products = np.dot(embeddings, query_emb)
        norms_emb = np.linalg.norm(embeddings, axis=1)
        norm_query = np.linalg.norm(query_emb)
        similarities = dot_products / (norms_emb * norm_query + 1e-9)
        
    scored_candidates = []
    
    # 2. Process Candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            
            # Layer 0: Honeypot filter
            if cid in honeypots:
                continue
                
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            
            # Layer 0: Hard JD filters
            # Make sure we check history titles and descriptions for CV/Speech/Robotics and Research filters
            history_texts = []
            for r in history:
                history_texts.append(r.get("title", ""))
                history_texts.append(r.get("description", ""))
            history_combined = " ".join(history_texts)
            
            combined_text = f"{headline} {summary} " + " ".join([s.get("name", "") for s in skills]) + " " + history_combined
            combined_lower = combined_text.lower()
            
            has_nlp_ir = any(kw in combined_lower for kw in NLP_IR_KEYWORDS)
            has_cv_speech = any(kw in combined_lower for kw in CV_SPEECH_KEYWORDS)
            
            # If they have CV/speech/robotics and lack NLP/IR, drop them
            if has_cv_speech and not has_nlp_ir:
                continue
                
            # Exclude pure research without production deployment
            has_research = any(kw in combined_lower for kw in RESEARCH_KEYWORDS)
            has_production = any(kw in combined_lower for kw in PRODUCTION_KEYWORDS)
            # If they have research keywords but NO production keywords, drop them
            if has_research and not has_production:
                continue
                
            # Layer 1: Scoring (Person-Job Fit)
            # Cosine similarity boost
            if similarities is not None and cid in cand_id_to_idx:
                cos_sim = float(similarities[cand_id_to_idx[cid]])
            else:
                cos_sim = get_fallback_similarity(cand)
                
            # Group by company to get true average company tenure (avoiding role promotion penalization)
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
                
            # Penalties: Consulting-only experience without product company
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
            
            # Additional penalty: LangChain/OpenAI only AI experience under 12 months (without pre-LLM ML foundations)
            ai_skills = [s for s in skills if any(kw in s.get("name", "").lower() for kw in ["langchain", "openai", "gpt-4", "llm", "rag"])]
            pre_llm_ml_skills = [s for s in skills if any(kw in s.get("name", "").lower() for kw in ["scikit-learn", "tensorflow", "pytorch", "nlp", "information retrieval", "vector search", "regression", "classification", "svm", "random forest", "xgboost", "clustering", "spacy", "nltk"])]
            if ai_skills and not pre_llm_ml_skills:
                if all(s.get("duration_months", 0) <= 12 for s in ai_skills):
                    fit_score *= 0.1
                    
            # Additional penalty: Senior management-only non-coding role in the last 18 months
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
                
            # Layer 2: Behavioral Calibration
            # notice_period_days <= 30 boost
            notice = signals.get("notice_period_days", 90)
            notice_mult = 1.15 if notice <= 30 else 1.0
            
            # recruiter_response_rate < 0.3 penalty
            resp_rate = signals.get("recruiter_response_rate", 0.0)
            resp_mult = 1.0
            if resp_rate < 0.3:
                resp_mult = 0.5
                
            # last_active_date > 6 months penalty
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
            
            # Extract additional signals for reasoning
            willing_relocate = signals.get("willing_to_relocate", False)
            loc = profile.get("location", "")
            
            scored_candidates.append({
                "candidate_id": cid,
                "score": calibrated_score,
                "yoe": profile.get("years_of_experience", 0.0),
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
    # Sort descending by score rounded to 4 decimal places, ascending by candidate_id to break ties
    scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    
    # 3. Reasoning Generation for Top 100
    top_100 = scored_candidates[:100]
    
    # Monotonic scores correction if needed
    # Ensure they are monotonically non-increasing (which they will be due to sorting)
    # Ranks range from 1 to 100
    output_rows = []
    for rank_idx, cand_info in enumerate(top_100):
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
        
        # Extract skills for template
        top_vector_db = extract_vector_db_skill(skills, headline, summary)
        top_eval_skill = extract_eval_skill(skills, headline, summary)
        
        # Inject honest concerns
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
            
        # Rank-dependent prefix/tone
        if rank_idx < 15:
            prefix = "Outstanding candidate: "
        elif rank_idx < 50:
            prefix = "Strong candidate: "
        else:
            prefix = "Decent adjacent fit: "
            
        # Select base structure based on rank index to vary templates
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
        
    # Write output to CSV
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
            
    print(f"Ranking complete. Wrote {len(output_rows)} rows to {args.out}.")

if __name__ == "__main__":
    main()
