import json
import os
from datetime import datetime

# Load honeypots
honeypots = {}
if os.path.exists("honeypots.json"):
    with open("honeypots.json", "r", encoding="utf-8") as f:
        honeypots = json.load(f)

# Define consulting/services companies
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "l&t", 
    "ltts", "ltimindtree", "deloitte", "pwc", "ey", "kpmg", "mphasis", 
    "hexaware", "ust global", "genpact", "cognizant technology solutions"
}

# Define AI/ML titles
ML_TITLES = {"ai", "ml", "machine learning", "nlp", "natural language", "data scientist", "data science", "applied scientist", "research engineer", "computer vision", "deep learning"}
ADJACENT_TITLES = {"backend", "software", "data engineer", "analytics engineer", "full stack", "programmer", "developer"}
DISQUALIFIED_TITLES = {"marketing", "hr", "human resources", "operations", "accountant", "finance", "sales", "graphic designer", "customer support", "mechanical", "civil"}

def score_candidate(cand):
    cid = cand["candidate_id"]
    if cid in honeypots:
        return 0.0, "Honeypot"
        
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # 1. Title Score
    current_title = profile.get("current_title", "").lower()
    title_score = 0.0
    
    # Check if current title matches ML
    is_ml_title = any(kw in current_title for kw in ML_TITLES)
    is_adjacent_title = any(kw in current_title for kw in ADJACENT_TITLES)
    is_disq_title = any(kw in current_title for kw in DISQUALIFIED_TITLES)
    
    # Check career history for ML titles
    has_ml_history = False
    for r in history:
        r_title = r.get("title", "").lower()
        if any(kw in r_title for kw in ML_TITLES):
            has_ml_history = True
            break
            
    if is_ml_title:
        title_score = 1.0
    elif is_adjacent_title:
        title_score = 0.6 if has_ml_history else 0.4
    elif is_disq_title:
        title_score = 0.0
    else:
        title_score = 0.2
        
    # 2. Experience Score (5-9 years is ideal, 6-8 is best)
    yoe = profile.get("years_of_experience", 0)
    if 6.0 <= yoe <= 8.0:
        exp_score = 1.0
    elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
        exp_score = 0.8
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 10.0:
        exp_score = 0.5
    elif 3.0 <= yoe < 4.0 or 10.0 < yoe <= 12.0:
        exp_score = 0.3
    else:
        exp_score = 0.1
        
    # 3. Company Type Score (Product vs Consulting)
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
        company_multiplier = 0.1  # Disqualified or heavily penalized
    elif not has_product:
        company_multiplier = 0.5
        
    # 4. Skills Score
    # We want NLP/IR and search/retrieval/embeddings skills
    # Penalize if they ONLY have CV/speech/robotics
    matching_skills = 0
    cv_speech_skills = 0
    nlp_search_skills = 0
    
    nlp_keywords = {"nlp", "natural language", "embeddings", "sentence-transformers", "transformer", "bert", "gpt", "llm", "fine-tuning", "rag", "retrieval", "search", "vector search", "milvus", "qdrant", "faiss", "pinecone", "weaviate", "elasticsearch", "opensearch", "semantic search", "indexing", "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map", "eval"}
    cv_speech_keywords = {"computer vision", "cv", "yolo", "object detection", "image classification", "gan", "cnn", "speech recognition", "tts", "robotics", "speech"}
    
    for s in skills:
        name = s.get("name", "").lower()
        # Check if match
        is_nlp = any(kw in name for kw in nlp_keywords)
        is_cv_speech = any(kw in name for kw in cv_speech_keywords)
        
        if is_nlp:
            nlp_search_skills += 1
        if is_cv_speech:
            cv_speech_skills += 1
            
    # Compute skill score
    skill_score = 0.0
    if nlp_search_skills > 0:
        skill_score = min(nlp_search_skills * 0.25, 1.0)
    elif cv_speech_skills > 0:
        skill_score = 0.2  # CV/Speech only is penalized
    else:
        skill_score = 0.1
        
    # Check for keyword stuffing:
    # If they have many AI skills but their title is non-technical, title_score is already 0.0
    
    # 5. Location Score
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)
    
    is_pune_noida = "pune" in loc or "noida" in loc
    is_india = "india" in country or "india" in loc
    
    tier_1_cities = {"bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "noida", "pune", "chennai", "hyderabad", "kolkata"}
    is_tier_1 = any(c in loc for c in tier_1_cities)
    
    if is_pune_noida:
        loc_score = 1.0
    elif is_tier_1:
        loc_score = 0.8
    elif is_india:
        loc_score = 0.6 if willing_relocate else 0.4
    else: # Outside India
        loc_score = 0.3 if willing_relocate else 0.1
        
    # Combine Fit Score
    fit_score = (title_score * 0.4 + exp_score * 0.3 + skill_score * 0.2 + loc_score * 0.1) * company_multiplier
    
    # 6. Behavioral Multiplier
    # Stated notice period: <= 30 days is best. 90+ days is penalized.
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_mult = 1.1
    elif notice <= 60:
        notice_mult = 1.0
    elif notice <= 90:
        notice_mult = 0.8
    else:
        notice_mult = 0.5
        
    # Last active date: target is 2026-06-24. Mismatch > 6 months (180 days) is heavily penalized
    last_active_str = signals.get("last_active_date", "")
    active_mult = 1.0
    if last_active_str:
        try:
            la_dt = datetime.strptime(last_active_str, "%Y-%m-%d")
            curr_dt = datetime(2026, 6, 24)
            days_inactive = (curr_dt - la_dt).days
            if days_inactive > 180:
                active_mult = 0.3  # Heavily down-weighted
            elif days_inactive > 90:
                active_mult = 0.7
        except Exception:
            pass
            
    # Recruiter response rate
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    if resp_rate >= 0.7:
        resp_mult = 1.1
    elif resp_rate >= 0.4:
        resp_mult = 1.0
    elif resp_rate >= 0.2:
        resp_mult = 0.7
    else:
        resp_mult = 0.3
        
    # Open to work
    otw = signals.get("open_to_work_flag", False)
    otw_mult = 1.1 if otw else 0.9
    
    # Connections and other activity (small boosters)
    github_score = signals.get("github_activity_score", -1)
    git_mult = 1.0
    if github_score > 50:
        git_mult = 1.05
    elif github_score == -1:
        git_mult = 0.95
        
    behavior_mult = notice_mult * active_mult * resp_mult * otw_mult * git_mult
    
    final_score = fit_score * behavior_mult
    
    return final_score, f"fit={fit_score:.3f}, title={title_score:.1f}, exp={exp_score:.1f}, skill={skill_score:.1f}, loc={loc_score:.1f}, comp={company_multiplier:.1f}, b_mult={behavior_mult:.3f}"

if __name__ == "__main__":
    candidates_scored = []
    
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            score, details = score_candidate(cand)
            if score > 0:
                candidates_scored.append((score, cand["candidate_id"], cand, details))
                
    # Sort descending
    candidates_scored.sort(key=lambda x: (-x[0], x[1]))
    
    print(f"Scored {len(candidates_scored)} non-honeypot candidates.")
    print("\nTop 15 candidates:")
    for idx, (score, cid, cand, details) in enumerate(candidates_scored[:15]):
        profile = cand["profile"]
        skills_str = ", ".join([s["name"] for s in cand.get("skills", [])[:5]])
        print(f"{idx+1}. ID: {cid}, Score: {score:.4f}, Title: {profile.get('current_title')}, YoE: {profile.get('years_of_experience')}, Location: {profile.get('location')}, Skills: {skills_str}")
        print(f"   Details: {details}")
