"""ROUGE evaluation module calculating ROUGE-1, ROUGE-2, and ROUGE-L."""

from __future__ import annotations


def compute_rouge_metrics(summary: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F-measure.

    Args:
        summary: The generated candidate summary.
        reference: The ground-truth reference summary.

    Returns:
        dict with keys: rouge1_p, rouge1_r, rouge1_f,
                        rouge2_p, rouge2_r, rouge2_f,
                        rougeL_p, rougeL_r, rougeL_f
    """
    clean_summary = summary.strip()
    clean_reference = reference.strip()

    if not clean_summary or not clean_reference:
        return {
            "rouge1_p": 0.0,
            "rouge1_r": 0.0,
            "rouge1_f": 0.0,
            "rouge2_p": 0.0,
            "rouge2_r": 0.0,
            "rouge2_f": 0.0,
            "rougeL_p": 0.0,
            "rougeL_r": 0.0,
            "rougeL_f": 0.0,
        }

    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'rouge-score'. Install with: pip install rouge-score"
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(clean_reference, clean_summary)

    return {
        "rouge1_p": round(scores["rouge1"].precision, 4),
        "rouge1_r": round(scores["rouge1"].recall, 4),
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_p": round(scores["rouge2"].precision, 4),
        "rouge2_r": round(scores["rouge2"].recall, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_p": round(scores["rougeL"].precision, 4),
        "rougeL_r": round(scores["rougeL"].recall, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }
