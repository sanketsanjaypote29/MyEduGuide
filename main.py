import streamlit as st
from PIL import Image
import pandas as pd
import streamlit.components.v1 as components
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from translate import Translator
import speech_recognition as sr
import os  # Import os for file operations
import base64  # Import base64 for encoding files

# Sample video data
video_data = pd.DataFrame({
    'Title': ['DSA', 'Video 2', 'Video 3'],
    'Description': ['Description 1', 'Description 2', 'Description 3'],
    'Category': ['Learn Coding', 'Learn Development', 'Prepare for University Exams'],
    'Thumbnail': [
        'https://images.unsplash.com/photo-1487088678257-3a541e6e3922?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8eW91dHViZSUyMHRodW1ibmFpbHxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60',
        'https://images.unsplash.com/photo-1487088678257-3a541e6e3922?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8eW91dHViZSUyMHRodW1ibmFpbHxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60',
        'https://images.unsplash.com/photo-1487088678257-3a541e6e3922?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8eW91dHViZSUyMHRodW1ibmFpbHxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60'
    ],
})

# Define functions for different pages
def page_home():
    st.title("Video Recommendation")
    st.write("Welcome to the Home Page!")
    search_query = st.text_input("", placeholder="search", key="search_query")
    selected_category = st.sidebar.selectbox("Select Category", video_data['Category'].unique(), key='category_option')

    if selected_category == "All Categories":
        filtered_data = video_data
    else:
        filtered_data = video_data[video_data['Category'] == selected_category]

    # Display recommended videos
    st.write("Recommended Videos")

    for index, row in filtered_data.iterrows():
        st.write(f"**{row['Title']}**")
        st.write(row['Description'])
        if st.button(f"Watch {row['Title']}", key=index):
            st.write(f"You clicked 'Watch {row['Title']}'")
            st.video("https://www.youtube.com/watch?v=-n2rVJE4vto&pp=ygUUc3RhY2sgZGF0YSBzdHJ1Y3R1cmU%3D")

    # Check if the user has entered a search query
    if search_query:
        # Call the recommend_educational_resources function to get recommendations
        recommended_resource = recommend_educational_resources(search_query)

        # Display the recommended educational resource
        display_educational_resource(recommended_resource)

# Function to recommend educational resources
def recommend_educational_resources(query):
    # Perform a web search or use APIs to find relevant educational resources
    # For simplicity, let's assume we're recommending a single resource for this example
    recommended_resource = {
        'Title': 'GeeksforGeeks - Introduction to Linked Lists',
        'Description': 'Learn about linked lists on GeeksforGeeks',
        'URL': 'https://www.geeksforgeeks.org/data-structures/linked-list/',
    }
    return recommended_resource

# Function to display educational resource recommendation
def display_educational_resource(resource):
    st.write("Recommended Educational Resource:")
    st.write(f"**Title:** {resource['Title']}")
    st.write(f"**Description:** {resource['Description']}")
    st.write(f"**URL:** [{resource['Title']}]({resource['URL']})")

def page_askAI():
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that I don't know, sorry for the inconvenience don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

    def set_custom_prompt():
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
        return prompt

    def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
        return qa_chain

    def load_llm():
        llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=10002,
        temperature=1.0
        )
        return llm

    def qa_bot():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa

    def final_result(query):
        qa_result = qa_bot()
        response = qa_result({'query': query})
        result = response.get('result')
        source = response.get('source_documents')
        return result, source

    def translate_text(text, lang):
    # Define the maximum character limit for translation (adjust as needed)
        max_char_limit = 500

    # Initialize an empty list to store translated segments
        translated_segments = []

    # Split the input text into segments based on the maximum character limit
        segments = [text[i:i + max_char_limit] for i in range(0, len(text), max_char_limit)]

    # Translate each segment individually
        for segment in segments:
            translator = Translator(to_lang=lang)
            translated_segment = translator.translate(segment)
            translated_segments.append(translated_segment)

    # Combine the translated segments into a single translated text
        translated_text = ' '.join(translated_segments)


    # Convert the audio to text using the speech recognition library
        try:
            user_input = r.recognize_google(audio)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

        # Call your QA function with the user's voice input
            response, sources = final_result(user_input)

            selected_language_code = language_codes[selected_language]

            translated_response = translate_text(response, selected_language_code)

            msg = {"role": "assistant", "content": translated_response}

            st.session_state.messages.append(msg)
            st.chat_message("assistant").write(msg["content"])

            

        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.write("Sorry, I encountered an error while processing the audio.")

# Continue with text input handling
    if prompt := st.chat_input("User Input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response, sources = final_result(prompt)

        if response not in response_download_links:
            selected_language_code = language_codes[selected_language]

            translated_response = translate_text(response, selected_language_code)

     

        # Display the response
            st.chat_message("assistant").write(translated_response)
        # st.text(translated_response)
            st.session_state.messages.append({"role": "assistant", "content": translated_response})

# Sidebar navigation
nav_option = st.sidebar.radio("Go to", ("Home", "Ask AI"))

# Clear the previous content
st.empty()

# Conditional content based on navigation option
if nav_option == "Home":
    page_home()
elif nav_option == "Ask AI":
    page_askAI()

