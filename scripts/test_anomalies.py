import json
from collections import Counter

def check_anomalies():
    counts = Counter()
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            history = cand.get("career_history", [])
            education = cand.get("education", [])
            skills = cand.get("skills", [])
            profile = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            
            signup = signals.get("signup_date")
            last_active = signals.get("last_active_date")
            if signup and last_active and signup > last_active:
                counts["signup_after_last_active"] += 1
                
            for e in education:
                sy = e.get("start_year")
                ey = e.get("end_year")
                if sy and ey and ey < sy:
                    counts["education_inverted"] += 1
                    
            yoe = profile.get("years_of_experience", 0)
            yoe_months = yoe * 12
            has_exceed = False
            for s in skills:
                dur = s.get("duration_months", 0)
                if dur > yoe_months + 12 and yoe > 0:
                    has_exceed = True
            if has_exceed:
                counts["skill_duration_exceeds_yoe"] += 1
                
            for r in history:
                start = r.get("start_date")
                if start and start > "2026-06-24":
                    counts["career_start_in_future"] += 1
                    break

    print("Counts of each anomaly:")
    for k, v in counts.items():
        print(f"{k}: {v}")
        
if __name__ == "__main__":
    check_anomalies()
