import gradio as gr
from faster_whisper import WhisperModel
import asyncio
import tempfile

import voice_service as vs
from AIVoiceAssistant import AIVoiceAssistant

# Initialize AI assistant and other constants
DEFAULT_MODEL_SIZE = "medium"
ai_assistant = AIVoiceAssistant()

# Core function to process the user's audio input
def process_audio(file):
    # The Gradio audio input returns a tuple (file_path, data)
    print(file)
    # Whisper model setup
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe the audio file
    transcription = asyncio.run(transcribe_audio(model, file))

    # Process the transcription and get a response from the AI
    response = asyncio.run(process_customer_input(transcription))
    
    # Convert AI response to audio using TTS
    audio_output = vs.text_to_speech(response)
    
    return audio_output

# Transcription function
async def transcribe_audio(model, file_path):
    try:
        segments, info = model.transcribe(file_path)
        transcription = ' '.join(segment.text for segment in segments)
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""

# Process customer input and interact with the assistant
async def process_customer_input(transcription):
    try:
        output = ai_assistant.interact_with_llm(transcription)
        return output
    except Exception as e:
        print(f"Error in processing input: {e}")
        return "Error processing request."

# Gradio interface setup with audio input and output
iface = gr.Interface(
    fn=process_audio, 
    #inputs=gr.Audio(),
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs="audio",
    title="Voice-Based Real Estate Assistant",
    description="Speak your queries about apartment details, towers, or ongoing projects.",
)

# Launch the Gradio app
iface.launch(share=True)