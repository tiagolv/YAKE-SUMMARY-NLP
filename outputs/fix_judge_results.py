import json
import re
from pathlib import Path

def extract_scores_from_raw(raw_response: str) -> dict:
    """Extract scores from raw_response using regex patterns."""
    scores = {}
    
    # Padrões para capturar "score": X (inteiro ou decimal)
    for dim in ["fidelity", "coverage", "coherence", "keyword_relevance"]:
        # Procura por "dim": { "score": valor
        pattern = rf'"{dim}"\s*:\s*\{{[^}}]*"score"\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            scores[dim] = float(match.group(1))
        else:
            scores[dim] = 3.0
    
    # Procura por "overall": valor
    overall_match = re.search(r'"overall"\s*:\s*(\d+(?:\.\d+)?)', raw_response)
    scores["overall"] = float(overall_match.group(1)) if overall_match else 3.0
    
    return scores

def main():
    input_path = Path("judge_results.json")
    if not input_path.exists():
        print("Ficheiro judge_results.json não encontrado.")
        return
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    fixed_count = 0
    for entry in data:
        raw = entry.get("judge", {}).get("raw_response", "")
        if not raw:
            continue
        
        # Se já tiver scores diferentes de 3, pode pular (mas vamos atualizar)
        extracted = extract_scores_from_raw(raw)
        
        # Atualiza os campos do judge
        entry["judge"]["fidelity"]["score"] = extracted.get("fidelity", 3)
        entry["judge"]["coverage"]["score"] = extracted.get("coverage", 3)
        entry["judge"]["coherence"]["score"] = extracted.get("coherence", 3)
        entry["judge"]["keyword_relevance"]["score"] = extracted.get("keyword_relevance", 3)
        entry["judge"]["overall"] = extracted.get("overall", 3)
        # Adiciona uma nota indicando que foi extraído por regex
        entry["judge"]["fidelity"]["justification"] = "Extracted from raw response (auto)"
        entry["judge"]["coverage"]["justification"] = "Extracted from raw response (auto)"
        entry["judge"]["coherence"]["justification"] = "Extracted from raw response (auto)"
        entry["judge"]["keyword_relevance"]["justification"] = "Extracted from raw response (auto)"
        fixed_count += 1
    
    # Guarda o ficheiro corrigido (substitui o original)
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Corrigidas {fixed_count} entradas em {input_path}")

if __name__ == "__main__":
    main()