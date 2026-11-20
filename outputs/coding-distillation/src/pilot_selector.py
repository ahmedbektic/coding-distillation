"""Identify and verify 20 problems from the frozen dataset with passing references."""
from pathlib import Path
from .common import jsonl, read_json, atomic_json
from .checkers import support_issues, invalid_test_indices
from .validate import DockerValidator
from .pipeline import load_config

def find_validated_problems(config_path="config-v2.yaml", target_count=20):
    config = load_config(config_path)
    validator = DockerValidator(config["validation"])
    problems = jsonl(Path(config["data_dir"]) / "problems/selected.jsonl")
    
    validated = []
    summary = []
    
    for i, prob in enumerate(problems):
        if len(validated) >= target_count:
            break
        if support_issues(prob) or invalid_test_indices(prob):
            continue
        ref_file = Path(config["results_dir"]) / "reference-sources" / f"{prob['id']}.json"
        if not ref_file.exists():
            continue
        refs = read_json(ref_file).get("references", [])
        if not refs:
            continue
        
        for ref in refs:
            v = validator.validate(ref["code"], prob)
            if v["accepted"]:
                validated.append(i)
                info = {
                    "pilot_index": len(validated),
                    "selection_index": i,
                    "problem_id": prob["id"],
                    "difficulty": prob["difficulty"],
                    "source": prob["source"],
                    "format": "callable" if prob["tests"]["fn_name"] else "stdio",
                    "fn_name": prob["tests"]["fn_name"],
                    "test_cases": len(prob["tests"]["inputs"]),
                    "reference_index": ref["reference_index"],
                    "validation_elapsed_seconds": v["elapsed_seconds"],
                }
                summary.append(info)
                print(f"Validated #{len(validated):02d}: index={i:03d} id={prob['id'][:10]} "
                      f"diff={prob['difficulty']:12s} src={prob['source']:10s} "
                      f"fmt={info['format']:8s} tests={info['test_cases']:3d}", flush=True)
                break
    
    out_path = Path(config["results_dir"]) / "pilot-validated-problems.json"
    atomic_json(out_path, summary)
    print(f"\nSaved {len(validated)} validated problem indices to {out_path}")
    return validated, summary

if __name__ == "__main__":
    find_validated_problems()
