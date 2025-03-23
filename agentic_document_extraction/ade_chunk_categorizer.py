from collections import defaultdict
from typing import Dict, List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

SECTION_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
            You are an expert document categorizer. You must assign this chunk of text to exactly **one** of the high-level document section categories. 

            ⚠️ Your job is to choose ONE AND ONLY ONE label from the following list (or create a new one ONLY if it's truly necessary):

                - General Info
                - Legal Terms
                - Pricing
                - Supplier Details
                - Document Information
                - Other

            Rules:
            - Pick only **one label** per text input.
            - Never invent multiple labels or a list of labels.
            - If nothing matches well, you may use "Other".

            Text:
            {text}

            Answer format: Just write a single label. No explanations. No punctuation.
            """
            )

def categorize_chunks_with_llm(llm, chunks: List[Dict]) -> List[Dict]:
    categorizer = SECTION_CLASSIFIER_PROMPT | llm
    labeled_chunks = []
    for chunk in chunks:
        try:
            response = categorizer.invoke({"text": chunk["text"]})
            label = response.content.strip()
        except Exception as e:
            print(f"⚠️ Failed to categorize chunk {chunk['chunk_id']}: {e}")
            label = "Unknown"
        chunk["section_label"] = label
        labeled_chunks.append(chunk)
    return labeled_chunks

def group_chunks_by_section(labeled_chunks: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for chunk in labeled_chunks:
        label = chunk.get("section_label", "Unknown")
        grouped[label].append(chunk)
    return dict(grouped)