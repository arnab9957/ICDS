import zipfile
import xml.etree.ElementTree as ET
import sys
import os

# Configure stdout to use UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def read_docx(file_path):
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # w:t elements contain the actual text in Word XML
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text_runs = []
            for elem in root.iter():
                # Check for w:t tag
                if elem.tag.endswith('}t'):
                    text_runs.append(elem.text or '')
                # Add newlines for paragraphs w:p
                elif elem.tag.endswith('}p'):
                    text_runs.append('\n')
            
            return "".join(text_runs).strip()
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_docx.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(read_docx(file_path))
