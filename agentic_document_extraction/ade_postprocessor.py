import json
import os
import re
import unicodedata
from typing import List, Dict
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from pathlib import Path


def load_ade_json(json_path: str) -> Dict:
    """Load structured ADE API JSON response."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("data", {})


def normalize_chunks(data: Dict) -> List[Dict]:
    """Flatten chunk and grounding info into a list of standardized entries."""
    return [
        {
            "chunk_id": c["chunk_id"],
            "page": g["page"],
            "chunk_type": c["chunk_type"],
            "text": c["text"].strip(),
            "bounding_box": g["box"]
        }
        for c in data.get("chunks", [])
        for g in c.get("grounding", [])
    ]


def extract_tables_from_chunks(chunks: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Extract HTML tables embedded in 'table' or 'form' chunk types."""
    tables = {}
    for c in chunks:
        if c["chunk_type"] in ["table", "form"]:
            soup = BeautifulSoup(c["text"], "html.parser")
            try:
                dfs = pd.read_html(StringIO(str(soup)))
                for i, df in enumerate(dfs):
                    key = f"{c['chunk_id']}_table{i}"
                    tables[key] = df
            except Exception as e:
                print(f"⚠️ Table parsing failed in {c['chunk_id']}: {e}")
    return tables


def summarize_chunks(chunks: List[Dict], model=None) -> Dict[str, str]:
    """Simple fallback summarizer via chunk type and page (no LLM)."""
    if model is None:
        summary = {}
        for c in chunks:
            key = f"{c['chunk_type']}_page{c['page']}"
            summary.setdefault(key, "")
            summary[key] += c["text"] + "\n\n"
        return summary
    else:
        raise NotImplementedError("🔧 LLM-based summarization not yet plugged into this method.")


def save_grouped_chunks(grouped: Dict[str, List[Dict]], output_dir: str):
    """Save full grouped chunk structure (JSON) and flattened preview (CSV)."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "grouped_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)

    rows = []
    for section, chunks in grouped.items():
        for chunk in chunks:
            rows.append({
                "section": section,
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text_preview": chunk["text"][:120].replace("\n", " ") + "..."
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "grouped_chunks.csv"), index=False)


def save_grouped_chunks_by_section(grouped_chunks: Dict[str, List[Dict]], output_dir: str):
    """Save each section's chunk texts into a Markdown file with sanitized filenames."""
    section_dir = Path(output_dir) / "grouped_sections"
    section_dir.mkdir(parents=True, exist_ok=True)

    for section, chunks in grouped_chunks.items():
        safe_section_name = re.sub(r'[^a-zA-Z0-9_-]', '_', section).strip("_")
        markdown_path = section_dir / f"{safe_section_name}.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(markdown_path, "w", encoding="utf-8") as md_file:
                for chunk in chunks:
                    md_file.write(f"### Chunk {chunk['chunk_id']} (Page {chunk['page']})\n\n")
                    md_file.write(chunk["text"] + "\n\n")
        except Exception as e:
            print(f"⚠️ Failed to write section '{section}' to file: {e}")


def clean_llm_summary(summary: str) -> str:
    """
    Clean LLM output by removing invisible Unicode characters and formatting artifacts.
    """
    # Remove HTML tags just in case
    summary = BeautifulSoup(summary, "lxml").get_text()

    # Replace known problematic characters
    summary = summary.replace("\u00a0", " ")  # non-breaking space
    summary = summary.replace("\u202f", " ")  # narrow non-breaking space
    summary = summary.replace("\u200b", "")   # zero-width space
    summary = summary.replace("\u2060", "")
    summary = summary.replace("\u2061", "")
    summary = summary.replace("\u2062", "")
    summary = summary.replace("\u2063", "")
    summary = summary.replace("\u2064", "")

    # Normalize all Unicode
    summary = unicodedata.normalize("NFKC", summary)

    # Strip control-formatting characters entirely
    summary = ''.join(ch for ch in summary if unicodedata.category(ch) != 'Cf')

    # Collapse redundant whitespace
    summary = re.sub(r'\s+', ' ', summary).strip()

    return summary


def summarize_grouped_chunks_with_llm(llm, grouped_chunks: Dict[str, List[Dict]]) -> Dict[str, str]:
    """Generate a GPT summary per section using all chunk texts."""
    from langchain.prompts import PromptTemplate

    summary_prompt = PromptTemplate.from_template(
        "Given the following document section:\n\n{text}\n\nWrite a concise paragraph summarizing its main content."
    )

    runnable = summary_prompt | llm
    section_summaries = {}

    for label, chunks in grouped_chunks.items():
        combined_text = "\n\n".join([c["text"] for c in chunks])
        try:
            result = runnable.invoke({"text": combined_text})
            raw_summary = getattr(result, "content", str(result)).strip()
            clean_summary = clean_llm_summary(raw_summary)
        except Exception as e:
            print(f"⚠️ Failed to summarize section '{label}': {e}")
            clean_summary = "[Summary unavailable due to error.]"
        section_summaries[label] = clean_summary

    return section_summaries