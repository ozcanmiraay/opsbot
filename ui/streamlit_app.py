import os
import sys
import streamlit as st
import pandas as pd
import tempfile

# Add opsbot-codebase/opsbot/ to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PARENT_DIR)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from langchain_gpt4o.schema_detector import infer_schema_from_chunks
from langchain_gpt4o.extractor import extract_records
from langchain_gpt4o.database_writer import write_to_db_and_csv

# ------------------- SETUP -------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="OpsBot PDF Extractor", layout="centered", page_icon="📄")
st.title("📄 OpsBot PDF Processing & Data Viewer")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("📎 Upload a PDF file to process", type=["pdf"])
doc_type = st.radio("Select document type:", options=["structured", "unstructured"])
run_pipeline = st.button("🔍 Run Extraction Pipeline")

if run_pipeline and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../langchain_gpt4o/outputs"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

    with st.status("Running extraction pipeline..."):
        st.write("📄 Loading PDF and splitting into chunks...")
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        st.write(f"🧠 Inferring schema from top chunks ({doc_type})...")
        schema_fields = infer_schema_from_chunks(llm, chunks, max_chunks=10, mode=doc_type)
        seen = set()
        schema_fields = [x for x in schema_fields if not (x in seen or seen.add(x))]
        st.success(f"✅ Inferred schema fields: {schema_fields}")

        st.write("🔍 Extracting structured records...")
        progress_bar = st.progress(0)
        records = []
        for idx, chunk in enumerate(chunks):
            partial_records = extract_records(llm, [(chunk, idx)], schema_fields)
            records.extend(partial_records)
            progress_bar.progress((idx + 1) / len(chunks))

        st.write("💾 Saving to DB and CSV...")
        write_to_db_and_csv(records, schema_fields, output_dir=OUTPUT_DIR)

        st.success("✅ Extraction pipeline completed successfully!")

    # ------------------- Display Output -------------------
    st.subheader("📊 Preview Extracted Records")
    if records:
        def normalize_df_for_streamlit(df):
            def clean_cell(value):
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value)
                elif isinstance(value, (dict, set)):
                    return str(value)
                elif pd.isna(value):
                    return ""
                else:
                    return str(value)
            return df.apply(lambda col: col.map(clean_cell))

        df = pd.DataFrame(records)
        df = normalize_df_for_streamlit(df)
        st.dataframe(df)
    else:
        st.warning("⚠️ No records were extracted from this file.")