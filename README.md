## Problem:
Customer support team often receives similar questions about room types, pricing, sea-facing penthouses, and layout images. Handling these repetitive queries takes up a lot of their time, making it challenging to manage all inquiries efficiently.

## Our Solution:
To ease the workload, we’re developing a customer support agent that answers common questions in real time. This intelligent agent will provide instant information on room types, amenities, floor plans, and pricing. If a customer shows interest, the system will automatically save their details for the team to follow up directly. We’ll focus on one tower, ensuring the chatbot can provide accurate and helpful information. Also, future plan is to roll this out across all towers and eventually add voice support as well.

## Libraries and Frameworks:
 - OpenAI: Used exclusively for generating responses to user queries through GPT-based models.
 - Streamlit: To build the web-based user interface for the chatbot, enabling real-time interaction.
 - Sentence Transformers (all-MiniLM-L6-v2): For creating sentence embeddings and efficient text comparison or querying tasks, which are essential for document search and retrieval.
 - FAISS: Used for efficient similarity search and nearest-neighbor search in embeddings space, allowing users to query and retrieve relevant information quickly.
 - PIL (Python Imaging Library): For handling image loading and processing.
 - docx: For reading and parsing Microsoft Word documents that contain detailed data about towers (floor plans, room details, etc.).
 - SpeechRecognition: For recognizing and processing speech input from users.
 - WhisperModel (from faster_whisper): For accurate transcription of voice inputs.
 - pyaudio: For handling live audio streams for real-time voice interaction.
 - voice_service as vs: A custom module for voice service processing.
 - Gradio: Will be considered for deploying the voice assistant, if needed, for interactive voice-based demonstrations.


![image](https://github.com/user-attachments/assets/90ec55b6-8c93-49ee-96bc-0a8814daae43)

## Objectives:
- Develop a customer support agent that reduces the workload by automating responses to frequently asked questions.
- Ensure the solution is easily accessible through a web chat interface.
- Create a structured data file containing all relevant information for the selected tower.
- Implement a system that captures customer interest for follow-up.
