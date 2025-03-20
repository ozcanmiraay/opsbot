# extractor.py

import json
from langchain.prompts import PromptTemplate

def extract_records(llm, indexed_chunks, schema_fields):
    prompt = PromptTemplate(
        template="""
You are a document extraction assistant. Given the following chunk from a document, extract a **single valid JSON object** with values for the schema fields below.

Schema fields (as keys):
{schema_fields}

Document chunk:
\"\"\"{doc_text}\"\"\"

Return only the JSON object. Do NOT include multiple rows, extra commentary, or explanations.
If a field is not found, use null.
""",
        input_variables=["schema_fields", "doc_text"]
    )

    chain = prompt | llm
    records = []

    for chunk, idx in indexed_chunks:
        try:
            print(f"🔍 Processing chunk {idx + 1}...")
            response = chain.invoke({
                "schema_fields": json.dumps(schema_fields),
                "doc_text": chunk.page_content
            })

            content = response.content.strip()
            content = content.replace("```json", "").replace("```python", "").replace("```", "").strip()

            record = json.loads(content)

            if not isinstance(record, dict):
                raise ValueError("Returned result is not a JSON object.")

            filled_fields = sum(1 for v in record.values() if v not in [None, "", "null"])
            if filled_fields >= 3:
                records.append(record)
            else:
                print(f"⚠️ Skipping chunk {idx + 1} — sparse record")

        except Exception as e:
            print(f"⚠️ Skipping chunk {idx + 1} due to error: {e}")
            continue

    return records