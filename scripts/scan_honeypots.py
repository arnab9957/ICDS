import json
import os
from datetime import datetime

def scan_honeypots(candidates_path, output_path):
    honeypots = {}
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            cid = cand["candidate_id"]
            
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            
            yoe = profile.get("years_of_experience", 0)
            total_yoe_months = yoe * 12.0
            
            reasons = []
            
            # Rule 1: Single role duration claimed exceeds total profile experience
            for r in history:
                dur = r.get("duration_months", 0)
                if dur > total_yoe_months + 0.5:
                    reasons.append(f"role_duration_exceeds_profile_total:claimed={dur},profile_total={total_yoe_months:.1f}")
                    break
                    
            # Rule 2: Claimed role duration differs significantly from start and end dates
            if not reasons:
                for r in history:
                    start_str = r.get("start_date")
                    end_str = r.get("end_date") or "2026-06-24"
                    duration_months = r.get("duration_months", 0)
                    try:
                        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                        actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                        if abs(actual_months - duration_months) > 6:
                            reasons.append(f"role_duration_date_inconsistency:actual={actual_months},claimed={duration_months}")
                            break
                    except Exception:
                        pass
                        
            # Rule 3: Extreme discrepancy between profile YoE and sum of all history durations
            if not reasons:
                total_history_months = sum(r.get("duration_months", 0) for r in history)
                if yoe > 5.0 and total_history_months < 24:
                    reasons.append(f"profile_yoe_exceeds_history_sum:yoe={yoe},history_months={total_history_months}")
                elif yoe < 3.0 and total_history_months > 96:
                    reasons.append(f"history_sum_exceeds_profile_yoe:yoe={yoe},history_months={total_history_months}")
                    
            # Rule 4: Claim 'expert' or 'advanced' proficiency but have 0 months of use
            if not reasons:
                expert_zero_dur = [s for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
                if len(expert_zero_dur) >= 3:
                    reasons.append(f"expert_skills_with_zero_duration:count={len(expert_zero_dur)}")
                    
            if reasons:
                honeypots[cid] = reasons
                
    # Ensure directory for output exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(honeypots, out, indent=2)
        
    print(f"Honeypot scan complete. Found {len(honeypots)} honeypots.")
    return honeypots

if __name__ == "__main__":
    scan_honeypots("data/candidates.jsonl", "data/honeypots.json")
