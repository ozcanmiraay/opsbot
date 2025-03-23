def extract_text_chunks(response_json):
    return [chunk["text"] for chunk in response_json.get("chunks", [])]

def extract_by_type(response_json, chunk_type="key_value"):
    return [chunk for chunk in response_json.get("chunks", []) if chunk.get("chunk_type") == chunk_type]

def save_raw_outputs(response_json, output_dir, base_filename="ade_output"):
    import os, json

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{base_filename}.json")
    md_path = os.path.join(output_dir, f"{base_filename}.md")

    with open(json_path, "w") as f:
        json.dump(response_json, f, indent=2)

    markdown_content = response_json.get("markdown", "")
    with open(md_path, "w") as f:
        f.write(markdown_content)

    return json_path, md_path