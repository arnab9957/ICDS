import json
from analyze_ranker import score_candidate
from datetime import datetime

def generate_reasoning(cand, score):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "")
    
    # Extract key matching skills
    nlp_keywords = {"nlp", "embeddings", "sentence-transformers", "transformer", "bert", "gpt", "llm", "fine-tuning", "rag", "retrieval", "search", "vector search", "milvus", "qdrant", "faiss", "pinecone", "weaviate", "elasticsearch", "opensearch", "semantic search", "indexing", "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map"}
    matched_skills = []
    for s in skills:
        name = s.get("name", "")
        if any(kw in name.lower() for kw in nlp_keywords):
            matched_skills.append(name)
            if len(matched_skills) >= 3:
                break
    
    # If no matched nlp skills, take top skills
    if not matched_skills:
        matched_skills = [s.get("name") for s in skills[:2]]
        
    skills_phrase = f"skills in {', '.join(matched_skills)}" if matched_skills else "strong ML background"
    
    # Companies
    companies = [r.get("company") for r in history if r.get("company")]
    if len(companies) >= 2:
        comp_phrase = f"experience at {companies[0]} and {companies[1]}"
    elif len(companies) == 1:
        comp_phrase = f"experience at {companies[0]}"
    else:
        comp_phrase = "applied ML experience"
        
    # Notice and response rate
    notice = signals.get("notice_period_days", 90)
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    
    # Location context
    loc_phrase = ""
    if "pune" in loc.lower() or "noida" in loc.lower():
        loc_phrase = f"based in target location {loc}"
    else:
        loc_phrase = f"located in {loc}"
        
    # Combine into sentences
    s1 = f"{title} with {yoe:.1f} years of experience, possessing {skills_phrase}."
    
    # Add details about company history and signals
    s2 = f"Demonstrates solid {comp_phrase} with high engagement ({resp_rate:.0%} response rate, {notice}d notice), {loc_phrase}."
    
    # Handle concerns
    concerns = []
    if notice >= 90:
        concerns.append(f"notice period is {notice} days")
    if not signals.get("willing_to_relocate", False) and "pune" not in loc.lower() and "noida" not in loc.lower():
        concerns.append("not willing to relocate")
        
    if concerns:
        s2 += f" Note: concern on {', '.join(concerns)}."
        
    return f"{s1} {s2}"

def print_top_15_reasonings():
    top_cands = []
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            score, details = score_candidate(cand)
            if score > 0:
                top_cands.append((score, cand))
                
    top_cands.sort(key=lambda x: -x[0])
    
    print("Top 15 candidates and their reasonings:")
    for idx, (score, cand) in enumerate(top_cands[:15]):
        cid = cand["candidate_id"]
        reason = generate_reasoning(cand, score)
        print(f"\n{idx+1}. {cid} (Score={score:.4f}):")
        print(f"   {reason}")

if __name__ == "__main__":
    print_top_15_reasonings()
