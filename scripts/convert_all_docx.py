import os
import zipfile
import xml.etree.ElementTree as ET

def convert_docx_to_txt(docx_path, txt_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            text_runs = []
            for elem in root.iter():
                if elem.tag.endswith('}t'):
                    text_runs.append(elem.text or '')
                elif elem.tag.endswith('}p'):
                    text_runs.append('\n')
            
            content = "".join(text_runs).strip()
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully converted {docx_path} -> {txt_path}")
    except Exception as e:
        print(f"Error converting {docx_path}: {e}")

if __name__ == "__main__":
    docx_files = [f for f in os.listdir('.') if f.endswith('.docx')]
    for docx_file in docx_files:
        txt_file = docx_file.replace('.docx', '.txt')
        convert_docx_to_txt(docx_file, txt_file)
