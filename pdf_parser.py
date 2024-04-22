from pypdf import PdfReader
from pathlib import Path
from pprint import pprint

def parse_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    pages = pdf_reader.pages

    full_text = ''
    for page in pages:
        full_text += page.extract_text()

    return full_text

if __name__ == '__main__':
    pdf_path = Path('./pdf/Tollerup_CV.pdf')
    full_text = parse_pdf(pdf_path)
    print(full_text)
   