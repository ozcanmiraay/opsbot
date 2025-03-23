import os
from .ade_client import send_document
from .ade_response_parser import save_raw_outputs

def run_agentic_pipeline(file_path: str, output_dir: str, is_pdf=True):
    print(f"📤 Sending file to ADE API: {file_path}")
    response = send_document(file_path, is_pdf=is_pdf)

    print("✅ Received response. Saving structured outputs...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    json_path, md_path = save_raw_outputs(response, output_dir, base_filename=base_name)

    print(f"📂 Saved: {json_path}")
    print(f"📂 Saved: {md_path}")
    return response