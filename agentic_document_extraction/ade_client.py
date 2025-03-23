import requests
from .ade_config import ADE_API_URL, HEADERS

def send_document(file_path: str, is_pdf=True, include_marginalia=True, include_metadata=True):
    files = {
        "pdf" if is_pdf else "image": open(file_path, "rb")
    }

    params = {
        "include_marginalia": str(include_marginalia).lower(),
        "include_metadata_in_markdown": str(include_metadata).lower()
    }

    response = requests.post(ADE_API_URL, headers=HEADERS, files=files, params=params)

    if response.status_code != 200:
        raise Exception(f"⚠️ ADE API error: {response.status_code} → {response.text}")
    
    return response.json()