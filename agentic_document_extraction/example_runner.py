import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agentic_document_extraction.ade_client import send_document
from agentic_document_extraction.ade_postprocessor import (
    load_ade_json,
    normalize_chunks,
    save_grouped_chunks,
    save_grouped_chunks_by_section,
    summarize_grouped_chunks_with_llm
)
from agentic_document_extraction.ade_chunk_categorizer import (
    categorize_chunks_with_llm,
    group_chunks_by_section
)

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Step 1 – Define PDF path
PDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mock_data", "proscia-ade-mock-document.pdf")
)
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"❌ PDF file not found: {PDF_PATH}")

print(f"\n📤 Sending file to ADE API: {PDF_PATH}")

# Step 2 – Send to ADE API and save response
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
json_output_path = os.path.join(OUTPUT_DIR, "proscia-ade-mock-document.json")

response_json = send_document(PDF_PATH)
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(response_json, f, indent=2)

print(f"✅ Received response. Saved structured output to: {json_output_path}")

# Step 3 – Postprocessing pipeline
data = load_ade_json(json_output_path)
normalized_chunks = normalize_chunks(data)
labeled_chunks = categorize_chunks_with_llm(llm, normalized_chunks)
grouped_sections = group_chunks_by_section(labeled_chunks)

# Step 4 – Display results
for label, chunks in grouped_sections.items():
    print(f"\n🗂️ {label} ({len(chunks)} chunks):")
    for c in chunks:
        print(f" - Chunk {c['chunk_id'][:8]} on page {c['page']}")

# Step 5 – Save results
save_grouped_chunks(grouped_sections, output_dir=OUTPUT_DIR)
save_grouped_chunks_by_section(grouped_sections, output_dir=OUTPUT_DIR)
print(f"\n✅ Grouped chunk data saved in: {OUTPUT_DIR}")

# Step 6 – Generate and save LLM summaries
section_summaries = summarize_grouped_chunks_with_llm(llm, grouped_sections)

summaries_md_path = os.path.join(OUTPUT_DIR, "section_summaries.md")
with open(summaries_md_path, "w", encoding="utf-8") as f:
    for section, summary in section_summaries.items():
        f.write(f"## {section}\n{summary}\n\n")

print(f"📝 Section summaries saved to: {summaries_md_path}")