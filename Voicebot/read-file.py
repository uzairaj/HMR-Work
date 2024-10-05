import docx

def read_word_file(file_path):
 
        doc = docx.Document(file_path)
        full_text = []

        for para in doc.paragraphs:
            full_text.append(para.text)
        
        return '\n'.join(full_text)

file_path = "/Users/uzairadamjee/Documents/Uzair/AI-Challenge/voice_assistant_llm-main/rag/H1.docx"
all_text = read_word_file(file_path)
print(all_text)