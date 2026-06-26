import os
import shutil

def organize():
    # Define directories
    dirs = ["docs", "data", "scripts"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
            
    # File mappings
    docs_files = [
        "README.docx",
        "README.txt",
        "job_description.docx",
        "job_description.txt",
        "redrob_signals_doc.docx",
        "redrob_signals_doc.txt",
        "submission_spec.docx",
        "submission_spec.txt",
        "candidate_schema.json"
    ]
    
    data_files = [
        "sample_candidates.json",
        "sample_submission.csv",
        "honeypots.json",
        "detected_honeypots.json"
    ]
    
    scripts_files = [
        "read_docx.py",
        "convert_all_docx.py",
        "scan_honeypots.py",
        "test_anomalies.py",
        "test_top30.py",
        "analyze_ranker.py"
    ]
    
    # Move function
    def move_files(files, target_dir):
        for f in files:
            if os.path.exists(f):
                try:
                    shutil.move(f, os.path.join(target_dir, f))
                    print(f"Moved {f} -> {target_dir}/")
                except Exception as e:
                    print(f"Error moving {f}: {e}")
                    
    move_files(docs_files, "docs")
    move_files(data_files, "data")
    move_files(scripts_files, "scripts")

if __name__ == "__main__":
    organize()
