import openai
from sentence_transformers import SentenceTransformer
import faiss
import os
from openai import OpenAI
import docx
from dotenv import load_dotenv
load_dotenv()

class AIVoiceAssistant:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini-2024-07-18"  # You can change this to GPT-4 if you want.gpt-3.5-turbo
        self._index = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._create_kb()
        self.chat_history = []  # Initialize chat history to maintain context across conversations

    def read_word_file(self, file_path):
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)

    def split_text_into_chunks(self, text):
        sentences = text.split('.')
        return sentences

    def _create_kb(self):
        try:
            file_path = "../Chatbot/data/H1.docx"
            all_text = self.read_word_file(file_path)
            text_chunks = self.split_text_into_chunks(all_text)
            # Generate embeddings for chunks
            self.embeddings = self.create_embeddings(text_chunks)
            # Create FAISS index
            self._index = self.create_faiss_index(self.embeddings)
            self.documents = text_chunks
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def create_embeddings(self, texts):
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().detach().numpy()

    def create_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search_kb(self, query):
        query_embedding = self.create_embeddings([query])
        D, I = self._index.search(query_embedding, k=3)
        return self.documents[I[0][0]]  # Return the most relevant chunk

    def interact_with_llm(self, customer_query):
        # Search the knowledge base for relevant context
        context = self.search_kb(customer_query)

        # Detailed prompt for the voice assistant
        detailed_prompt = """
        You are a professional real estate voice assistant for HMR Waterfront, responsible for providing information regarding its towers, apartments, and their associated room details.
        - If a query is about a specific tower, respond with tower information with available residential apartments listed in it.
        - If a query is about a specific apartment, respond with the corresponding room types and their specific View, Floor Range, and Total Assigned Area details.
        - If a query is about a specific room type, provide its respective information.
        - If a query is too broad, provide general information first and ask a follow-up question to narrow down the user's request.
        - If no specific tower, apartment, or room type is mentioned, ask a follow-up question to clarify the user’s needs.
        - If you don't know the answer, simply say "I don't know" without making up information.
        - Ensure that responses are clear, structured, and user-friendly.
        """

        # Include the chat history in the prompt for context
        messages = [{"role": "system", "content": detailed_prompt}]
        for message in self.chat_history:
            messages.append(message)
        
        # Add the user's current query
        messages.append({"role": "user", "content": f"Context:{customer_query}"})
        print('--------')
        print(messages)
        print('--------')
        # Call OpenAI API
        response = OpenAI().chat.completions.create(
            model=self.model,
            messages=messages
        )
     
        answer = response.choices[0].message.content
        print(answer)
        # Store the conversation in chat history
        self.chat_history.append({"role": "user", "content": customer_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer