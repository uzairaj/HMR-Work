## Problem:
The customer support team often receives similar questions about room types, pricing, sea-facing penthouses, and layout images. Handling these repetitive queries takes up a lot of their time, making it challenging to manage all inquiries efficiently.

## Our Solution:
To ease the workload, we’re developing a customer support agent that answers common questions in real time. This intelligent agent will provide instant information on room types, amenities, floor plans, and pricing. If a customer shows interest, the system will automatically save their details for the team to follow up directly. We’ll focus on one tower, ensuring the chatbot can provide accurate and helpful information.

## Libraries and Frameworks:
 - OpenAI: Used exclusively for generating responses to user queries through GPT-based models.
 - Streamlit: To build the web-based user interface for the chatbot, enabling real-time interaction.
 - Sentence Transformers (all-MiniLM-L6-v2): For creating sentence embeddings and efficient text comparison or querying tasks, which are essential for document search and retrieval.
 - FAISS: Used for efficient similarity search and nearest-neighbor search in embeddings space, allowing users to query and retrieve relevant information quickly.
 - PIL (Python Imaging Library): For handling image loading and processing.
 - docx: For reading and parsing Microsoft Word documents that contain detailed data about towers (floor plans, room details, etc.).
 - SpeechRecognition: For recognizing and processing speech input from users.
 - WhisperModel (from faster_whisper): For accurate transcription of voice inputs.
 - Gradio: Deploying the voice assistan app.

## Architecture:
![image](https://github.com/user-attachments/assets/90ec55b6-8c93-49ee-96bc-0a8814daae43)

## Getting Started:

- Clone the Repository.
- Install the Required Dependencies: Make sure you have pip installed. Then run:
```
pip install -r requirements.txt
```
- Set Up the OpenAI API Key: Create a .env file in the root directory and add your key as follows:
```
OPENAI_API_KEY="sk-"
```

# Running Chatbot:

```
# File Structure

HMR-Work/
├── requirements.txt          # Project dependencies
├── .env                      # Environment file for storing API key
└── README.md

HMR-Work/Chatbot
│
├── hmr-bot.py                    # Main application file (Streamlit interface)
```
Run the Application: Use the following command to start the Streamlit application:
```
streamlit run hmr-bot.py
```
The application will start running locally, and you can access it by opening the URL provided by Streamlit.

# Running Voicebot:

```
# File Structure

HMR-Work/
├── requirements.txt          # Project dependencies
├── .env                      # Environment file for storing API key
└── README.md

HMR-Work/Voicebot
│
├── app.py                    # Main application file (Gradio interface)
├── AIVoiceAssistant.py
├── voice_service.py 
```
Run the Application: 
```
python3 app.py
```

## Future Enhancements:
- The project is designed to be scalable and adaptable. Allowing easy expansion to other towers in the future.
- Handle different data source type, as each tower got their own format in form of images, pdf files.
- Saving information of users that are interested in the projects, mean potential leads.


## Objectives:
- Develop a customer support agent that reduces the workload by automating responses to frequently asked questions.
- Ensure the solution is easily accessible through a web chat interface.
- Create a structured data file containing all relevant information for the selected tower.
- Implement a system that captures customer interest for follow-up.
