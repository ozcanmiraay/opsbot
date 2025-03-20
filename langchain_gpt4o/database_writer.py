import os
import sqlite3
import pandas as pd

def normalize_record(record, schema_fields):
    return {key: str(record.get(key) or "").strip() for key in schema_fields}

def write_to_db_and_csv(records, schema_fields, output_dir, db_name="records.db", csv_name="records.csv"):
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, db_name)
    csv_path = os.path.join(output_dir, csv_name)

    # Normalize data
    normalized = [normalize_record(rec, schema_fields) for rec in records]

    # Save to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 🔥 Drop table first to avoid schema mismatch errors
    cursor.execute("DROP TABLE IF EXISTS extracted_data")

    # Recreate table with current schema
    quoted_columns = ", ".join([f'"{field}" TEXT' for field in schema_fields])
    cursor.execute(f'CREATE TABLE extracted_data (id INTEGER PRIMARY KEY AUTOINCREMENT, {quoted_columns})')
    conn.commit()

    for rec in normalized:
        quoted_field_names = ", ".join([f'"{field}"' for field in schema_fields])
        placeholders = ", ".join(["?"] * len(schema_fields))
        values = tuple(rec.get(f, "") for f in schema_fields)  # FIXED LINE
        cursor.execute(f"INSERT INTO extracted_data ({quoted_field_names}) VALUES ({placeholders})", values)

    conn.commit()
    conn.close()

    # Save to CSV
    df = pd.DataFrame(normalized)
    df.to_csv(csv_path, index=False)

    print(f"✅ DB written to: {db_path}")
    print(f"✅ CSV exported to: {csv_path}")