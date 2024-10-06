from dotenv import load_dotenv
import os
import json
import base64
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import openai
from openai import OpenAI
import docx

from PIL import Image
import pytesseract
import re
# Load environment variables from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
# Load your pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set directories
output_dir = 'dump/'
index_file = os.path.join(output_dir, 'faiss_index.bin')
file_map_file = os.path.join(output_dir, 'file_map.json')
word_file =  'data/H1.docx'
image_dir= 'data/layout/'

### Extract image location from files 
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def parse_filename(filename):
    # Check for special room types like PH (Penthouse) and TH (Townhouse)
    match = re.match(r'h(\d+)-(PH\d*|TH\d*|\d+[A-Za-z]\d*)', filename)
    if match:
        tower_name = f"H{match.group(1)}"
        room_code = match.group(2)
        return tower_name, room_code
    
    return None, None, None
def parse_image_text(image_text):
    view = re.search(r'View\s*:\s*(.*)', image_text)
    floor_range = re.search(r'Floor Range\s*:\s*(.*)', image_text)
    unit_net_area = re.search(r'Unit Net Aera\s*=\s*(.*)', image_text)
    common_area = re.search(r'Common Area\s*=\s*(.*)', image_text)
    sellable_area = re.search(r'Sellable Area\s*=\s*(.*)', image_text)
    assigned_parking_bay = re.search(r'Assigned Parking Bay\s*=\s*(.*)', image_text)
    total_assigned_area = re.search(r'Total Assigned Area\s*=\s*(.*)', image_text)

    view = view.group(1).strip() if view else None
    floor_range = floor_range.group(1).strip() if floor_range else None
    unit_net_area = unit_net_area.group(1).strip() if unit_net_area else None
    common_area = common_area.group(1).strip() if common_area else None
    sellable_area = sellable_area.group(1).strip() if sellable_area else None
    assigned_parking_bay = assigned_parking_bay.group(1).strip() if assigned_parking_bay else None
    total_assigned_area = total_assigned_area.group(1).strip() if total_assigned_area else None
    
    return view, floor_range, unit_net_area, common_area, sellable_area, assigned_parking_bay, total_assigned_area

def extract_text_and_images(images_dir):
    data = {'tower_name': None, 'rooms': []}
    base_dir = os.getcwd()
    # Check if the images_dir is a valid relative path; if not, use it as a direct path
    images_full_path = os.path.join(base_dir, images_dir)
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(images_dir, filename)
            tower_name, room_type = parse_filename(filename)            
            data['tower_name'] = tower_name
            image_text = extract_text_from_image(image_path)
            view, floor_range, unit_net_area, common_area, sellable_area, assigned_parking_bay, total_assigned_area = parse_image_text(image_text)
            room_details = {
                'room_type': room_type,
                'view': view,
                'floor_range': floor_range,
                'unit_net_area': unit_net_area,
                'common_area': common_area,
                'sellable_area': sellable_area,
                'assigned_parking_bay': assigned_parking_bay,
                'total_assigned_area': total_assigned_area,
                'layout': image_path 
            }
            data['rooms'].append(room_details)
    
    return data

def process_all_images(image_dir):
    
    # extracted_data = extract_text_and_images(image_dir)
    output_file = os.path.join(output_dir, "processed_images.json")
    # with open(output_file, 'w') as json_file:
    #     json.dump(extracted_data, json_file)
    # output_file = os.path.join(output_dir, "processed_images.json")
    
    # Check if the JSON file already exists
    if not os.path.isfile(output_file):
        extracted_data = extract_text_and_images(image_dir)  # Assuming this function is defined elsewhere
        with open(output_file, 'w') as json_file:
            json.dump(extracted_data, json_file)
        


def read_word_file(file_path):
 
    doc = docx.Document(file_path)
    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)
    
    return '\n'.join(full_text)


def create_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().detach().numpy()


def index_data(output_dir):
    file_map = []
    all_text = read_word_file(word_file)
    # Split the text into smaller chunks (e.g., by sentences or paragraphs)
    text_chunks = split_text_into_chunks(all_text)
    # Create embeddings for each chunk
    embeddings = create_embeddings(text_chunks)
    
    # Update the file_map to reference each chunk separately
    file_map.extend([('H1.docx', chunk) for chunk in text_chunks])
    print("Embeddings shape:", embeddings.shape)

    # Ensure that embeddings are properly shaped
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    print("Index embedding shape:", index.d)
    
    # Add embeddings to the FAISS index
    index.add(embeddings)
    return index, file_map


def split_text_into_chunks(text):
    sentences = text.split('.-')
    chunks = []
    
    for sentence in sentences:
        chunks.append(sentence)

    return chunks

def load_index_and_file_map():
    index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
    with open(file_map_file, 'r') as f:
        file_map = json.load(f)
    return index, file_map

def save_index_and_file_map(index, file_map):
    faiss.write_index(index, index_file)
    with open(file_map_file, 'w') as f:
        json.dump(file_map, f)

def query_index(query, index, file_map):
    query_embedding = create_embeddings([query])
    D, I = index.search(query_embedding, k=7)
    retrieved_texts = [(file_map[i][0], file_map[i][1]) for i in I[0]]
    return retrieved_texts

def stream_generate_response(query, index, file_map):
    #context=data
    retrieved_texts = query_index(query, index, file_map)
    print(retrieved_texts)
    context = "\n".join(text for _, text in retrieved_texts)
    
    detailed_prompt= """Act as a real estate virtual assistant designed to provide information about HMR Waterfront towers, apartments and their associated room details.
    - If a query is about a specific tower, response with tower information with available residential apartments list in it.
    - If a query is about a specific apartment, respond with the corresponding room types, and their specific View, Floor Range, Total Assigned Area details in bullet points.
    - If a query is about a specific room type, only then dislay all information (such as View, Floor Range, all Area sizes etc).
    - If a query is too broad, provide general information first and ask a follow-up question to narrow down the user's request.    
    - If no specific tower or apartment or room type is mentioned, ask a follow-up question to clarify the user’s needs.
    - Ensure that the responses are clear, structured, and user-friendly, avoiding technical.
    - Display layout images only when the user explicitly asks for them or agrees to see them after being prompted.
    """
    #Use the prompt in the API call
    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": detailed_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ]
    )
    print('-----------------')
    print(response)
    collected_messages = ""
    chunk_message = response.choices[0].message.content
    collected_messages += chunk_message
    return collected_messages



def query_data(query,index, file_map, output_dir):
    response='' 
    file_path = os.path.join(output_dir, "processed_images.json")
    data={}
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    if 'layout' in query.lower():
        for room in data['rooms']:
            if room['room_type'].lower() in query.lower():
                print('Layout: ', room['layout'])
                layout_image_path = room['layout']
                #response += f"\nLayout image for room {room['room_type']} in {data['tower_name']} found. Image path: {layout_image_path}."
                img = Image.open(layout_image_path)
                st.image(img)
                break  # Exit after displaying the specific layout
    else:
        response = stream_generate_response(query, index, file_map)
    return response

# Handling index and data loading/saving
if not os.path.exists(index_file) or not os.path.exists(file_map_file):
    process_all_images(image_dir)
    index, file_map = index_data(output_dir)
    save_index_and_file_map(index, file_map)
else:
    index, file_map = load_index_and_file_map()

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("hmr-img.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center;
    position: relative;
}}

[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.9);  /* Darkens the image (adjust for desired darkness) */
    z-index: 0;
}}

.stApp {{
    z-index: 1;
}}

h1 {{
    color: white;
    text-align: center;
    font-size: 50px;
}}

/* Lighter color for chat message text */
.stChatMessage {{
    color: #e0e0e0;  /* Light grey text for better contrast */
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
}}

.stTextInput input {{
    width: 600px;
    padding: 15px;
    font-size: 18px;
    border: 2px solid #ffffff;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    color: white;
    margin-top: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}}

.stButton button {{
    width: 150px;
    padding: 12px;
    background-color: #1E90FF;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
}}

.stButton button:hover {{
    background-color: #4682B4;
}}

</style>
"""

# Injecting the background CSS and custom input box styling
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.markdown("<h1>HMR Bot</h1>", unsafe_allow_html=True)

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize chat history only if not already set

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='stChatMessage'>{message['content']}</div>", unsafe_allow_html=True)

# Input field for user's message
user_prompt = st.chat_input("Ask...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(f"<div class='stChatMessage'>{user_prompt}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Send user's message to GPT-4o and get a response
    response = query_data(user_prompt, index, file_map, output_dir)
    assistant_response = response

    # Store the assistant's response in chat history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display GPT-4o's response
    with st.chat_message("assistant"):
        st.markdown(f"<div class='stChatMessage'>{assistant_response}</div>", unsafe_allow_html=True)
