import os
import sqlite3
import pandas as pd

def sanitize_field(field):
    return field.strip().lower().replace(" ", "_").replace("-", "_")

def normalize_record(record, schema_fields):
    return {sanitize_field(key): str(record.get(key) or "").strip() for key in schema_fields}

def write_to_db_and_csv(records, schema_fields, output_dir, db_name="records.db", csv_name="records.csv"):
    os.makedirs(output_dir, exist_ok=True)

    if not records:
        print("⚠️ No records to save. Skipping database and CSV export.")
        return None, None

    db_path = os.path.join(output_dir, db_name)
    csv_path = os.path.join(output_dir, csv_name)

    # Normalize field names and data
    clean_fields = [sanitize_field(f) for f in schema_fields]
    normalized = [normalize_record(rec, schema_fields) for rec in records]

    # Save to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop and recreate table
    cursor.execute("DROP TABLE IF EXISTS extracted_data")
    quoted_columns = ", ".join([f'"{field}" TEXT' for field in clean_fields])
    cursor.execute(f'CREATE TABLE extracted_data (id INTEGER PRIMARY KEY AUTOINCREMENT, {quoted_columns})')
    conn.commit()

    for rec in normalized:
        quoted_field_names = ", ".join([f'"{field}"' for field in clean_fields])
        placeholders = ", ".join(["?"] * len(clean_fields))
        values = tuple(rec.get(f, "") for f in clean_fields)
        cursor.execute(f"INSERT INTO extracted_data ({quoted_field_names}) VALUES ({placeholders})", values)

    conn.commit()
    conn.close()

    # Save to CSV
    df = pd.DataFrame(normalized)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✅ DB written to: {db_path}")
    print(f"✅ CSV exported to: {csv_path}")

    return db_path, csv_path