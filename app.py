import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader, 
                                                  CSVLoader, 
                                                  PyPDFLoader, 
                                                  TextLoader)


VALID_FILE_TYPES = [
    'Website', 'Youtube', 'Pdf', 'Csv', 'Txt'
]

MODEL_CONFIG = {'Groq': 
                    {'models': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                     'chat': ChatGroq},
                'OpenAI': 
                    {'models': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
                     'chat': ChatOpenAI}}

MEMORY = ConversationBufferMemory()

# Loaders
def load_website(url):
    """
    Loads and retrieves the content from a given website URL.

    This function uses a WebBaseLoader to load the content from a website and
    returns the extracted content as a single string, with page contents 
    concatenated using two newline characters ('\n\n').

    Args:
        url (str): The URL of the website to load.

    Returns:
        str: The concatenated content of the loaded website.
    """
    loader = WebBaseLoader(url)
    documents_list = loader.load()
    document = '\n\n'.join([doc.page_content for doc in documents_list])
    return document

def load_youtube(video_id):
    """
    Loads and retrieves the transcript content from a YouTube video.

    This function uses a YoutubeLoader to load the transcript of the specified 
    YouTube video (in Portuguese by default). The video transcript is 
    concatenated into a single string, with individual parts separated by 
    two newline characters ('\n\n').

    Args:
        video_id (str): The unique identifier of the YouTube video.

    Returns:
        str: The concatenated transcript content of the YouTube video.
    """
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    documents_list = loader.load()
    document = '\n\n'.join([doc.page_content for doc in documents_list])
    return document

def load_csv(file_path):
    """
    Loads and retrieves the content from a CSV file.

    This function uses a CSVLoader to load the content of the specified CSV file.
    The content of each document (i.e., each row or section of the CSV) is 
    concatenated into a single string, with each part separated by two newline 
    characters ('\n\n').

    Args:
        file_path (str): The file path to the CSV file to be loaded.

    Returns:
        str: The concatenated content of the CSV file.
    """
    loader = CSVLoader(file_path)
    documents_list = loader.load()
    document = '\n\n'.join([doc.page_content for doc in documents_list])
    return document

def load_pdf(file_path):
    """
    Loads and retrieves the content from a PDF file.

    This function uses a PyPDFLoader to load the content of the specified PDF file.
    The content of each page in the PDF is concatenated into a single string, 
    with each page's content separated by two newline characters ('\n\n').

    Args:
        file_path (str): The file path to the PDF file to be loaded.

    Returns:
        str: The concatenated content of the PDF file.
    """
    loader = PyPDFLoader(file_path)
    documents_list = loader.load()
    document = '\n\n'.join([doc.page_content for doc in documents_list])
    return document

def load_txt(file_path):
    """
    Loads and retrieves the content from a plain text file.

    This function uses a TextLoader to load the content of the specified text file.
    The content of each part of the text file is concatenated into a single string, 
    with each part separated by two newline characters ('\n\n').

    Args:
        file_path (str): The file path to the text file to be loaded.

    Returns:
        str: The concatenated content of the text file.
    """
    loader = TextLoader(file_path)
    documents_list = loader.load()
    document = '\n\n'.join([doc.page_content for doc in documents_list])
    return document

def load_files(file_type, file):
    """
    Loads and retrieves content from a file based on its type.

    This function determines the type of the file (Website, Youtube, Pdf, Csv, or Txt)
    and calls the appropriate loader function to extract and return the content.
    For file types that require uploading (Pdf, Csv, Txt), the file is temporarily 
    saved using NamedTemporaryFile before being loaded.

    Args:
        file_type (str): The type of the file ('Website', 'Youtube', 'Pdf', 'Csv', 'Txt').
        file (str or file-like object): The file path or file object to be loaded.
            For 'Website' and 'Youtube', this is a URL. For 'Pdf', 'Csv', and 'Txt', 
            this is an uploaded file object.

    Returns:
        str: The concatenated content of the loaded file based on its type.
    """
    if file_type == 'Website':
        document = load_website(file)
    if file_type == 'Youtube':
        document = load_youtube(file)
    if file_type == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(file.read())
            temp_name = temp.name
        document = load_pdf(temp_name)
    if file_type == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(file.read())
            temp_name = temp.name
        document = load_csv(temp_name)
    if file_type == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(file.read())
            temp_name = temp.name
        document = load_txt(temp_name)
    return document

def load_model(provider, model, api_key, file_type, file):
    """
    Loads a model and initializes a conversational chain based on a provided document.

    This function loads the content of a file using the `load_files` function and
    sets up a chat model based on the specified provider and model. A system message 
    is generated based on the document content, and a conversational chain is created 
    using a chat prompt template. The chain is then stored in the Streamlit session state.

    Args:
        provider (str): The provider of the chat model (e.g., 'Groq', 'OpenAI').
        model (str): The specific model to use (e.g., 'gpt-4o-mini', 'llama-3.1-70b-versatile').
        api_key (str): The API key required to access the model.
        file_type (str): The type of the file ('Website', 'Youtube', 'Pdf', 'Csv', 'Txt').
        file (str or file-like object): The file path or file object to be loaded.

    Returns:
        None: The function updates the session state with the initialized conversational chain.
    """
    document = load_files(file_type, file)

    system_message = '''You are a friendly assistant named Oracle.
    You have access to the following information from a document of type {}: 

    ####
    {}
    ####

    Use the provided information as a basis for your responses.

    Whenever you encounter $ in your output, replace it with S.

    If the document contains something like "Just a moment...Enable JavaScript and cookies to continue"
    suggest the user reload the Oracle!'''.format(file_type, document)
    
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = MODEL_CONFIG[provider]['chat'](model=model, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def chat_page():
    """
    Displays the main chat interface for interacting with the Oracle.

    This function creates a chat interface in Streamlit where users can 
    interact with the Oracle. It retrieves the conversational chain from 
    the session state. If the chain is not initialized, an error message is 
    shown prompting the user to load the Oracle. The function also manages 
    memory, loading previous chat history and updating it with new messages 
    from both the user and the AI.

    The interface includes:
    - A chat header with a welcome message.
    - A chat input box for user interaction.
    - Display of the conversation history using a buffer memory.
    - Streaming responses from the AI model based on the user's input.

    Args:
        None

    Returns:
        None: The function directly updates the Streamlit interface and session state.
    """
    st.header('🤖Welcome to the Oracle', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Load the Oracle')
        st.stop()

    memory = st.session_state.get('memory', MEMORY)
    for message in memory.buffer_as_messages:
        chat = st.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input('Talk to the Oracle')
    if user_input:
        chat = st.chat_message('human')
        chat.markdown(user_input)

        chat = st.chat_message('ai')
        response = chat.write_stream(chain.stream({
            'input': user_input, 
            'chat_history': memory.buffer_as_messages
            }))
        
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        st.session_state['memory'] = memory

def sidebar():
    """
    Displays the sidebar interface for file upload and model selection in Streamlit.

    This function creates a two-tab interface in the sidebar:
    
    1. **File Upload**: Allows users to select and upload files or input URLs depending on 
       the selected file type (Website, Youtube, Pdf, Csv, Txt).
       - Website: Input a website URL.
       - Youtube: Input a YouTube video URL.
       - Pdf: Upload a PDF file.
       - Csv: Upload a CSV file.
       - Txt: Upload a text file.

    2. **Model Selection**: Lets the user select a model provider (e.g., 'Groq', 'OpenAI'), 
       choose a specific model, and input the corresponding API key.

    Additionally, the sidebar provides buttons to initialize the Oracle with the selected 
    model and file content, and to clear the conversation history.

    Args:
        None

    Returns:
        None: The function directly updates the Streamlit sidebar interface and session state.
    """
    tabs = st.tabs(['File Upload', 'Model Selection'])
    with tabs[0]:
        file_type = st.selectbox('Select file type', VALID_FILE_TYPES)
        if file_type == 'Website':
            file = st.text_input('Enter the website URL')
        if file_type == 'Youtube':
            file = st.text_input('Enter the video URL')
        if file_type == 'Pdf':
            file = st.file_uploader('Upload a PDF file', type=['.pdf'])
        if file_type == 'Csv':
            file = st.file_uploader('Upload a CSV file', type=['.csv'])
        if file_type == 'Txt':
            file = st.file_uploader('Upload a TXT file', type=['.txt'])
    with tabs[1]:
        provider = st.selectbox('Select model provider', MODEL_CONFIG.keys())
        model = st.selectbox('Select model', MODEL_CONFIG[provider]['models'])
        api_key = st.text_input(
            f'Enter the API key for {provider}',
            value=st.session_state.get(f'api_key_{provider}'))

        st.session_state[f'api_key_{provider}'] = api_key
    
    if st.button('Initialize Oracle', use_container_width=True):
        load_model(provider, model, api_key, file_type, file)
    if st.button('Clear Conversation History', use_container_width=True):
        st.session_state['memory'] = MEMORY

def main():
    with st.sidebar:
        sidebar()
    chat_page()

if __name__ == '__main__':
    main()
