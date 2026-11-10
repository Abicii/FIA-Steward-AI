"""
Synthetic Evaluation Dataset Generator.

Generates realistic user and steward questions from markdown regulation chunks
using an LLM (LiteLLM / OpenAI / Gemini).

Usage:
    python scripts/generate_synthetic_eval.py --input "parsed data/FIA 2027 F1 Regulations...md" --output tests/data/synthetic_eval_set.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.services.chunking import chunk_document

SYSTEM_PROMPT = """You are an expert FIA Formula 1 Race Steward and regulatory analyst.
Given an excerpt from the FIA Regulations, write realistic, challenging test questions that an F1 team principal, race engineer, or steward would ask during a race weekend.
For each question, specify:
1. "query": The natural question asked in realistic language (may use racing jargon like SC, VSC, pit lane, Parc Fermé, gearbox, etc.).
2. "expected_keywords": Key regulatory phrases or article numbers required in the answer.
3. "category": e.g., "Pit Lane & Safety", "Penalties", "Parc Ferme", "Tyres", "Power Unit", "Race Procedures".
Output JSON only with a list of question objects.
"""


def generate_questions_for_chunk(chunk_content: str, metadata: dict[str, Any], questions_per_chunk: int = 2) -> list[dict[str, Any]]:
    """
    Calls an LLM to generate synthetic test queries for a given chunk.
    Uses LiteLLM (which supports OpenAI, Gemini, Claude, Ollama, etc.).
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm is not installed. Install it with: pip install litellm")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: No LLM API key detected (OPENAI_API_KEY, GEMINI_API_KEY).")
        return []

    model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")

    prompt = f"""Article: {metadata.get('citation', '')}
Section Content:
{chunk_content[:2000]}

Generate {questions_per_chunk} distinct questions whose answers are directly found in this text.
Return in JSON format:
[
  {{"query": "...", "category": "...", "expected_keywords": ["..."]}}
]"""

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
        questions = data if isinstance(data, list) else data.get("questions", [])
        for q in questions:
            q["expected_article"] = metadata.get("article_number", "")
            q["expected_sub_section"] = metadata.get("sub_section", "")
            q["source"] = metadata.get("source", "")
        return questions
    except Exception as exc:
        print(f"Failed to parse LLM response: {exc}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset from regulations markdown")
    parser.add_argument("--input", type=Path, required=True, help="Path to cleaned .md file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/synthetic_eval_set.json"),
        help="Target output JSON path",
    )
    parser.add_argument("--sample-chunks", type=int, default=15, help="Number of chunks to sample")
    args = parser.parse_args()

    content = args.input.read_text(encoding="utf-8")
    chunks = chunk_document(content, filename=args.input.name)

    print(f"Loaded {len(chunks)} chunks from {args.input.name}.")
    # Filter to substantive chunks with citations
    valid_chunks = [c for c in chunks if c.metadata.get("article_number") and c.metadata.get("article_number") != "0"]
    step = max(1, len(valid_chunks) // args.sample_chunks)
    sampled = valid_chunks[::step][: args.sample_chunks]

    all_questions = []
    print(f"Sampling {len(sampled)} chunks across different articles to generate questions...")
    for idx, c in enumerate(sampled, start=1):
        print(f"Generating for [{idx}/{len(sampled)}] {c.metadata.get('citation')}...")
        qs = generate_questions_for_chunk(c.page_content, c.metadata)
        all_questions.extend(qs)

    if all_questions:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2)
        print(f"Successfully generated {len(all_questions)} questions saved to {args.output}")
    else:
        print("No questions generated. Ensure an LLM API key is exported if using dynamic generation.")


if __name__ == "__main__":
    main()
