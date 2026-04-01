import pandas as pd
from PyPDF2 import PdfReader

def process_single_file(file_obj, kb, filename=None, full_path=None):
    """
    Orchestrates the ingestion of different file types into the KnowledgeBase.
    
    Args:
        file_obj: The file-like object or path.
        kb: The KnowledgeBase instance to populate.
        filename (str): Optional name of the file.
        full_path (str): Optional absolute path to the file.
        
    Returns:
        str: The name of the processed file.
    """
    fname = filename if filename else getattr(file_obj, 'name', 'unknown_file')
    
    # 1. PDF Handler: Iterates through pages and extracts raw text string.
    if fname.endswith(".pdf"):
        reader = PdfReader(file_obj)
        for i, page in enumerate(reader.pages):
            p_text = page.extract_text() or ""
            # Handover to KB: Text is cleaned and chunked within the KnowledgeBase class.
            kb.process_text(fname, p_text, i + 1, full_path=full_path)
            
    # 2. Tabular Handler (CSV): Converts rows to semantic strings.
    elif fname.endswith(".csv"):
        df = pd.read_csv(file_obj)
        kb.process_dataset(fname, df)
        
    # 3. Tabular Handler (Excel): Supports both .xls and .xlsx formats.
    elif fname.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_obj)
        kb.process_dataset(fname, df)
        
    # 4. Text/Markdown Handler: Reads full content as a single semantic unit (initially).
    elif fname.endswith((".md", ".txt")):
        if hasattr(file_obj, 'read'):
            content = file_obj.read().decode("utf-8")
        else:
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        kb.process_text(fname, content, 1, full_path=full_path)
        
    return fname
