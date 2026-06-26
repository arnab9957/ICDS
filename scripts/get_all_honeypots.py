import json
from datetime import datetime

def is_honeypot(cand):
    cid = cand["candidate_id"]
    history = cand.get("career_history", [])
    education = cand.get("education", [])
    skills = cand.get("skills", [])
    profile = cand.get("profile", {})
    
    # 1. Skill anomaly: expert/advanced skills with 0 duration
    expert_zero_dur = [s for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
    if len(expert_zero_dur) >= 3:
        return True, f"skills_zero_dur:{len(expert_zero_dur)}"
        
    # 2. profile yoe mismatch with history years
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(r.get("duration_months", 0) for r in history)
    history_years = total_months / 12.0
    if yoe > 0:
        if yoe > 5.0 and history_years < 2.0:
            return True, f"yoe_mismatch_low:yoe={yoe},hist={history_years:.2f}"
        elif yoe < 3.0 and history_years > 8.0:
            return True, f"yoe_mismatch_high:yoe={yoe},hist={history_years:.2f}"
            
    # 3. career history date vs duration mismatch
    for r in history:
        start_str = r.get("start_date")
        end_str = r.get("end_date") or "2026-06-24"
        duration_months = r.get("duration_months", 0)
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d")
            actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if abs(actual_months - duration_months) > 6:
                return True, f"duration_mismatch:actual={actual_months},claimed={duration_months}"
        except Exception:
            pass
            
    # 4. Education end_year < start_year
    for e in education:
        sy = e.get("start_year")
        ey = e.get("end_year")
        if sy and ey and ey < sy:
            return True, f"edu_dates:{sy}>{ey}"
            
    # 5. multiple current roles at different companies
    current_roles = [r for r in history if r.get("is_current")]
    if len(current_roles) >= 2:
        companies = set(r.get("company", "").strip().lower() for r in current_roles)
        if len(companies) >= 2:
            return True, f"multiple_current:{list(companies)}"
            
    return False, ""

if __name__ == "__main__":
    honeypots = {}
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            flag, reason = is_honeypot(cand)
            if flag:
                honeypots[cand["candidate_id"]] = reason
                
    print(f"Total honeypots found: {len(honeypots)}")
    with open("honeypots.json", "w", encoding="utf-8") as out:
        json.dump(honeypots, out, indent=2)
