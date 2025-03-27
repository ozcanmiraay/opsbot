# Full Streamlit app with improved design for demo purposes

import streamlit as st
import os
import re
import json
import tempfile
import pandas as pd
from bs4 import BeautifulSoup
import io
from pathlib import Path
from dotenv import load_dotenv
import time
from itertools import cycle
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI

from agentic_document_extraction.ade_client import send_document
from agentic_document_extraction.ade_postprocessor import (
    load_ade_json,
    normalize_chunks,
    deduplicate_chunks,
    save_grouped_chunks,
    save_grouped_chunks_by_section,
    summarize_grouped_chunks_with_llm,
    clean_llm_summary 
)
from agentic_document_extraction.ade_chunk_categorizer import (
    categorize_chunks_with_llm,
    group_chunks_by_section
)

# ────────────────────────────── SETUP ──────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# ───────────────────── PAGE CONFIG ─────────────────────
st.set_page_config(page_title="ADE-Powered DocIQ", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
            color: #f1f1f1;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 15px;
        }
        th, td {
            border: 1px solid #444;
            padding: 8px 12px;
            text-align: left;
            color: #f8f8f8;
        }
        th {
            background-color: #1f1f1f;
            font-weight: bold;
        }
        td {
            background-color: #2b2b2b;
        }
        tr:nth-child(even) td {
            background-color: #232323;
        }
        .block-container {
            padding: 2rem 4rem;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #3A78C2;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ───────────────────── HEADER ─────────────────────
st.title("📄 ADE-Powered Document Intelligence")
st.caption("Built for Proscia | Extract, structure, and summarize unstructured PDFs — in seconds.")
st.markdown("""
This tool leverages **LLM-enhanced document intelligence** to:
- Automatically extract and categorize semantic chunks from complex documents
- Visualize extracted sections in a readable format
- Generate concise, AI-driven summaries per section

Ideal for processing **contracts, clinical protocols, reports, and more**.
""")

# ───────────────────── UPLOAD ─────────────────────
st.subheader("📤 Upload Your Document")
uploaded_file = st.file_uploader(
    "Drop a PDF (max 2 pages / 50MB). ADE will extract and categorize its content for structured analysis.",
    type=["pdf"]
)

# ───────────────────── RENDER CHUNK CONTENT ─────────────────────
def render_chunk_content(content, chunk_id=None):
    if "<table" in content and "</table>" in content:
        header_text = content.split("<table")[0].strip()
        table_html = "<table" + content.split("<table")[1]

        if header_text:
            st.markdown(header_text, unsafe_allow_html=True)

        row_count = table_html.count("<tr>")
        dynamic_height = min(800, max(200, 50 + row_count * 40))
        components.html(f"""
            <html><head><style>
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #444;
                padding: 8px 12px;
                text-align: left;
                color: #f1f1f1;
            }}
            th {{
                background-color: #1f1f1f;
            }}
            td {{
                background-color: #2b2b2b;
            }}
            tr:nth-child(even) td {{
                background-color: #232323;
            }}
            </style></head>
            <body>{table_html}</body></html>
        """, height=dynamic_height, scrolling=True)

        try:
            soup = BeautifulSoup(table_html, "lxml")
            rows = soup.find_all("tr")
            data = []
            for row in rows:
                cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
                data.append(cols)

            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                df = pd.DataFrame(data)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="⬇️ Export This Table as CSV",
                data=csv_data,
                file_name=f"chunk_{chunk_id or 'table'}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.warning(f"⚠️ CSV export failed: {e}")

    elif content.strip().startswith("|") or re.search(r"\|\s*\w", content):
        st.markdown(content, unsafe_allow_html=True)
    elif "\t" in content:
        lines = content.strip().split("\n")
        rows = [line.split("\t") for line in lines]
        html = "<table><thead><tr>" + "".join(f"<th>{cell}</th>" for cell in rows[0]) + "</tr></thead><tbody>"
        for row in rows[1:]:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        html += "</tbody></table>"

        components.html(f"""
            <html><head><style>
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #444;
                padding: 8px 12px;
                text-align: left;
                color: #f1f1f1;
            }}
            th {{
                background-color: #1f1f1f;
            }}
            td {{
                background-color: #2b2b2b;
            }}
            tr:nth-child(even) td {{
                background-color: #232323;
            }}
            </style></head>
            <body>{html}</body></html>
        """, height=min(800, max(200, 50 + len(rows) * 40)), scrolling=True)

        try:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Export This Table as CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name=f"chunk_{chunk_id or 'tsv'}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"⚠️ TSV export failed: {e}")
    else:
        st.markdown(content, unsafe_allow_html=True)

# ───────────────────── SESSION INIT ─────────────────────
if "summaries" not in st.session_state:
    st.session_state.summaries = None
if "grouped_sections" not in st.session_state:
    st.session_state.grouped_sections = None
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None

# ───────────────────── ADE PIPELINE ─────────────────────
@st.cache_data(show_spinner=False)
def run_ade_pipeline(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    output_dir = Path("agentic_document_extraction/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ade_output_temp.json"

    tips = cycle([
        "🔍 Analyzing document layout...",
        "📊 Detecting tables and figures...",
        "🧾 Parsing form fields...",
        "🧠 Identifying semantic chunks...",
        "💡 Preparing structured content for LLM categorization..."
    ])
    st.info("📡 Preparing your document...")
    placeholder = st.empty()
    for _ in range(4):
        placeholder.info(next(tips))
        time.sleep(1.5)

    with st.spinner("🚀 Sending document to ADE API and awaiting response... (~1–2 min)"):
        response_json = send_document(pdf_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2)

    data = load_ade_json(json_path)
    normalized = normalize_chunks(data)
    normalized = deduplicate_chunks(normalized)
    labeled = categorize_chunks_with_llm(llm, normalized)
    grouped = group_chunks_by_section(labeled)

    save_grouped_chunks(grouped, output_dir=str(output_dir))
    save_grouped_chunks_by_section(grouped, output_dir=str(output_dir))

    return grouped, output_dir

# ───────────────────── MAIN ─────────────────────
if uploaded_file:
    st.markdown("### ⏳ Processing your document...")
    grouped_sections, output_dir = run_ade_pipeline(uploaded_file.read())
    st.success("✅ Document processing complete!")

    st.session_state.grouped_sections = grouped_sections
    st.session_state.output_dir = output_dir
    st.session_state.summaries = None

# ───────────────────── DISPLAY RESULTS ─────────────────────
if st.session_state.grouped_sections:
    st.header("🧩 Explore Document Structure")
    st.markdown("Select a section from the dropdown to see extracted chunks and their page context.")

    section_names = list(st.session_state.grouped_sections.keys())
    selected_section = st.selectbox("📂 Select a section to view its chunks", section_names, key="section_selector")

    section_chunks = st.session_state.grouped_sections[selected_section]
    st.subheader(f"🧠 Chunks in Section: {selected_section}")
    for i, chunk in enumerate(section_chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        page = chunk.get("page", "N/A")
        content = chunk.get("text", "[No text found]")

        with st.expander(f"🔸 Chunk {chunk_id} (Page {page})", expanded=False):
            render_chunk_content(content, chunk_id=chunk_id)

    st.markdown("---")
    st.subheader("📝 AI-Powered Section Summaries")
    st.markdown("Generate concise summaries for each section using GPT-4o. Useful for reviews, audits, and handoffs.")

    if st.button("✨ Generate AI Summary for All Sections"):
        with st.spinner("💬 Running GPT summarization on grouped content..."):
            st.session_state.summaries = summarize_grouped_chunks_with_llm(
                llm, st.session_state.grouped_sections
            )

    if st.session_state.summaries:
        st.subheader("📌 Section Summaries")
        for section, summary in st.session_state.summaries.items():
            st.markdown(f"### 🔹 {section}")
            st.write(clean_llm_summary(summary))

    st.markdown("---")
    st.success(f"📂 All results saved in: `{st.session_state.output_dir}`")

# Footer
st.markdown("---")
st.markdown("📘 _Built with ❤️ by [Miray Ozcan](https://www.linkedin.com/in/miray-ozcan/) | Powered by LangChain, OpenAI & Streamlit_")