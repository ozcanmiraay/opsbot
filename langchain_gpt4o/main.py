import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from schema_detector import infer_schema_from_chunks
from extractor import extract_records
from database_writer import write_to_db_and_csv

# ------------------- USAGE INSTRUCTION -------------------
def print_usage():
    print("""
Usage:
    python main.py <pdf_filename>

Arguments:
    <pdf_filename>   Name of the PDF inside 'mock_data' folder (e.g., mock_contract_1.pdf)

Example:
    python main.py mock_contract_1.pdf
    """)

# ------------------- ARGUMENT PARSING -------------------
if len(sys.argv) != 2:
    print_usage()
    sys.exit(1)

pdf_filename = sys.argv[1]

# ------------------- SETUP -------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY missing in .env")

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

BASE_DIR = os.path.dirname(__file__)
MOCK_DATA_DIR = os.path.join(BASE_DIR, "../mock_data")
PDF_PATH = os.path.join(MOCK_DATA_DIR, pdf_filename)

if not os.path.exists(PDF_PATH):
    print(f"❌ PDF file not found at: {PDF_PATH}")
    sys.exit(1)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- LOAD & SPLIT -------------------
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# ------------------- INFER SCHEMA -------------------
print("🧠 Inferring schema using top chunks...")
schema_fields = infer_schema_from_chunks(llm, chunks, max_chunks=10)

# Deduplicate fields
seen = set()
schema_fields = [x for x in schema_fields if not (x in seen or seen.add(x))]
print(f"✅ Final inferred schema: {schema_fields}")

# ------------------- EXTRACT DATA -------------------
print("🔍 Extracting structured records...")
records = extract_records(llm, chunks, schema_fields)

# ------------------- STORE RESULTS -------------------
write_to_db_and_csv(records, schema_fields, output_dir=OUTPUT_DIR)