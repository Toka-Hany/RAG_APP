import re

def extract_numbers(text):
    """
    Extract numbers exactly as written.
    """
    pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
    return re.findall(pattern, text)