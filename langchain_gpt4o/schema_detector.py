# schema_detector.py

import json
import re
import difflib
from langchain.prompts import PromptTemplate

# Customizable list of allowed field themes for unstructured mode
FILTERED_KEYWORDS = [
    "name", "id", "date", "dob", "company", "title", "department", "salary",
    "skills", "experience", "email", "phone", "city", "address", "platform",
    "workflow", "diagnostic", "user", "contact", "feature"
]

def contains_allowed_keyword(field):
    """Check if a field contains any allowed keyword (used in unstructured mode)."""
    field_lower = field.lower()
    return any(keyword in field_lower for keyword in FILTERED_KEYWORDS)

def normalize_field_name(field):
    """Convert field name to normalized snake_case lowercase for semantic comparison."""
    field = re.sub(r'[^a-zA-Z0-9\s]', '', field)  # remove special chars
    field = re.sub(r'\s+', '_', field)  # replace spaces with underscores
    return field.lower().strip()

def deduplicate_semantic_fields(fields, similarity_threshold=0.85):
    """Consolidate semantically similar field names using fuzzy matching and normalization."""
    normalized_map = {}
    canonical_fields = []

    for field in fields:
        norm = normalize_field_name(field)
        best_match = None
        for existing in canonical_fields:
            existing_norm = normalize_field_name(existing)
            similarity = difflib.SequenceMatcher(None, norm, existing_norm).ratio()
            if similarity >= similarity_threshold:
                best_match = existing
                break
        if best_match:
            continue  # Already clustered under a canonical field
        canonical_fields.append(field)
        normalized_map[norm] = field

    return canonical_fields

def infer_schema_from_chunks(llm, chunks, max_chunks=10, mode="unstructured"):
    """Infer and clean schema fields from top chunks using LLM and filtering."""
    prompt = PromptTemplate(
        template="""
You are a smart assistant helping design a database schema for structured data extraction.

Your goal is to return a concise list of field names to represent the structure of a database table for the information in the document.

🟢 Instructions:
- Use canonical field names (e.g., "date_of_birth", not "dob" or "DateOfBirth").
- Do not include multiple synonyms for the same field — pick only one standard name.
- Avoid vague or overly specific field names.
- Group semantically similar concepts under a single representative field name.
- Prefer snake_case lowercase naming convention.

Document chunk:
\"\"\"{doc_text}\"\"\"

Return ONLY a JSON array of field names like:
["field_1", "field_2", "field_3"]

Do NOT include any explanations or extra commentary.
""",
        input_variables=["doc_text"]
    )

    chain = prompt | llm
    inferred_fields = []

    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f"🧠 Inferring schema from chunk {i+1}...")
        try:
            response = chain.invoke({"doc_text": chunk.page_content})
            raw_output = response.content.strip()
            cleaned = raw_output.replace("```json", "").replace("```python", "").replace("```", "").strip()
            fields = json.loads(cleaned)
            if isinstance(fields, list):
                inferred_fields.extend(fields)
        except Exception as e:
            print(f"⚠️ Schema inference failed on chunk {i+1}: {e}")

    # Deduplicate exact matches
    seen = set()
    deduped = [f for f in inferred_fields if not (f in seen or seen.add(f))]

    # Apply keyword filtering (optional)
    if mode == "unstructured":
        deduped = [f for f in deduped if contains_allowed_keyword(f)]

    # Semantic deduplication to avoid DoB vs DateOfBirth etc.
    deduped = deduplicate_semantic_fields(deduped)

    return deduped