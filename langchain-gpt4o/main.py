import os
import json
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ----------------------------- SETUP ---------------------------------- #
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

OUTPUT_DIR = "/Users/mirayozcan/Desktop/opsbot-codebase/opsbot/langchain-gpt4o/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ LOAD & CHUNK DOCUMENT ------------------------ #
loader = PyPDFLoader("mock_data/employee_info_mock_data.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(pages)

# ------------------------ LLM & PROMPT SETUP --------------------------- #
llm = ChatOpenAI(model="gpt-4", api_key=api_key)

prompt_template = """
You are an intelligent document parser. Extract and return a JSON dictionary with the following employee fields from the given text block.

Text Block:
\"\"\"{doc_text}\"\"\"

Extract and return the following keys:
- Name
- Employee Id
- City
- Department
- Company
- Skills
- Experience
- Dob
- Salary

If any field is missing, return null for that field.
Ensure the output is valid JSON only without extra commentary.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["doc_text"])
chain = prompt | llm

# ------------------------- RUN CHAIN + VALIDATE ------------------------ #
employee_data = []

for idx, chunk in enumerate(chunks):
    print(f"\n📄 Processing chunk {idx + 1}...")
    try:
        response = chain.invoke({"doc_text": chunk.page_content})
        content = response.content.strip()

        # Clean leading triple backticks or json tags if LLM added them
        content = content.strip("`").replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(content)
            employee_data.append(data)
        except json.JSONDecodeError as e:
            print("⚠️ JSON parsing failed:", e)
            print("Response returned by LLM:\n", content)
            continue

    except Exception as e:
        print(f"❌ Error in LLM processing (chunk {idx + 1}):", e)
        continue

# ------------------------- NORMALIZE FIELDS ------------------------ #
def normalize_employee_record(record):
    def normalize_str(value):
        return str(value).strip() if value is not None else ""

    def normalize_salary(value):
        if isinstance(value, (int, float)):
            return f"${value:,.2f}"
        value_str = str(value).strip().replace(",", "")
        if not value_str.startswith("$"):
            value_str = f"${value_str}"
        return value_str

    def normalize_skills(value):
        if isinstance(value, list):
            return ", ".join(value)
        elif isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value.replace("'", '"'))
                    if isinstance(parsed, list):
                        return ", ".join(parsed)
                except:
                    return value
        return value

    return {
        "Name": normalize_str(record.get("Name")),
        "Employee Id": normalize_str(record.get("Employee Id")),
        "City": normalize_str(record.get("City")),
        "Department": normalize_str(record.get("Department")),
        "Company": normalize_str(record.get("Company")),
        "Skills": normalize_skills(record.get("Skills")),
        "Experience": normalize_str(record.get("Experience")),
        "Dob": normalize_str(record.get("Dob")),
        "Salary": normalize_salary(record.get("Salary"))
    }

normalized_records = [normalize_employee_record(rec) for rec in employee_data]

# ---------------------- STORE RESULTS TO DATABASE ---------------------- #
db_path = os.path.join(OUTPUT_DIR, "employee_records.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    employee_id TEXT,
    city TEXT,
    department TEXT,
    company TEXT,
    skills TEXT,
    experience TEXT,
    dob TEXT,
    salary TEXT
)
""")
conn.commit()

for record in normalized_records:
    cursor.execute("""
        INSERT INTO employees (name, employee_id, city, department, company, skills, experience, dob, salary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["Name"],
        record["Employee Id"],
        record["City"],
        record["Department"],
        record["Company"],
        record["Skills"],
        record["Experience"],
        record["Dob"],
        record["Salary"]
    ))

conn.commit()
conn.close()
print(f"\n✅ {len(normalized_records)} records inserted into {db_path}")

# ---------------------- CSV EXPORT ---------------------- #
csv_path = os.path.join(OUTPUT_DIR, "employee_records.csv")
df = pd.DataFrame(normalized_records)
df.to_csv(csv_path, index=False)
print(f"✅ Exported normalized results to {csv_path}")