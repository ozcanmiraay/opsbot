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

# ------------------- Session State Init -------------------
for key in ["inferred_schema", "chunks", "custom_fields", "selected_fields"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("📎 Upload a PDF file to process", type=["pdf"])
if uploaded_file and st.button("🔍 Run Extraction Pipeline"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("📄 Loading and chunking the PDF..."):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

    with st.spinner("🧠 Inferring schema..."):
        inferred_schema = infer_schema_from_chunks(llm, chunks, max_chunks=10)
        inferred_schema = list(dict.fromkeys(inferred_schema))  # deduplicate

    st.session_state["chunks"] = chunks
    st.session_state["inferred_schema"] = inferred_schema
    st.session_state["custom_fields"] = []
    st.session_state["selected_fields"] = inferred_schema.copy()

# ------------------- Customize Schema -------------------
if st.session_state["inferred_schema"]:
    st.subheader("🧠 Step 1: Inferred Schema")
    st.success("✅ Inference complete!")

    st.subheader("🛠 Step 2: Customize Schema")

    # Custom Field Adder
    custom_input = st.text_input("➕ Add any custom fields (comma-separated):")
    if st.button("➕ Add Custom Fields"):
        new_fields = [f.strip() for f in custom_input.split(",") if f.strip()]
        updated = False
        for field in new_fields:
            if field not in st.session_state["custom_fields"]:
                st.session_state["custom_fields"].append(field)
                updated = True
            if field not in st.session_state["selected_fields"]:
                st.session_state["selected_fields"].append(field)
                updated = True
        if updated:
            st.rerun()

    # Merge custom and inferred schema
    all_fields = list(dict.fromkeys(
        st.session_state["inferred_schema"] + st.session_state["custom_fields"]
    ))

    # ✅ Avoid direct key conflict, assign to var instead
    selected_fields_ui = st.multiselect(
        "✔️ Select fields to keep:",
        options=all_fields,
        default=st.session_state["selected_fields"]
    )

    # ✅ Sync selection only when changed
    if selected_fields_ui != st.session_state["selected_fields"]:
        st.session_state["selected_fields"] = selected_fields_ui
        st.rerun()

    if not st.session_state["selected_fields"]:
        st.warning("⚠️ Please select or add at least one field to proceed.")
    else:
        if st.button("🚀 Run Data Extraction", key="run_extraction_button"):
            OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../langchain_gpt4o/outputs"))
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
            with st.status("Running extraction pipeline..."):
                records = []
                progress_bar = st.progress(0)
                for idx, chunk in enumerate(st.session_state["chunks"]):
                    partial_records = extract_records(llm, [(chunk, idx)], st.session_state["selected_fields"])
                    records.extend(partial_records)
                    progress_bar.progress((idx + 1) / len(st.session_state["chunks"]))

                write_to_db_and_csv(records, st.session_state["selected_fields"], output_dir=OUTPUT_DIR)
                st.success("✅ Extraction completed!")

            st.subheader("📊 Extracted Data Preview")
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
                st.warning("⚠️ No records were extracted.")

# Footer
st.markdown("---")
st.markdown("🔧 Built with 💙 by **Miray Ozcan** | Powered by **LangChain + GPT-4o + Streamlit**")