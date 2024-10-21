# Oracle Chat Application

## Description

This project is an interactive chat application built with `Streamlit` and `Langchain`, allowing interaction with different AI models (such as Groq and OpenAI) to answer questions and provide information based on files uploaded by the user. The project can load and process content from **websites**, **YouTube videos**, **PDF files**, **CSV files**, and **text files**.

The main goal is to provide an AI-based assistant that can interpret information from various data sources and use that information to answer questions contextually.

## Features

- **File Upload**: Loads content from websites, YouTube videos, PDF, CSV, and text files.
- **AI Interaction**: Integrates with AI models from Groq and OpenAI to generate intelligent responses based on the uploaded content.
- **Conversation History**: Keeps track of conversation history to provide a continuous dialogue experience.
- **Graphical Interface**: Uses Streamlit to provide an interactive graphical interface.

## Installation

### Prerequisites

- **Python 3.7+** must be installed on your system.
- A `requirements.txt` file containing the project dependencies.

### Installation Steps

1. Clone this repository into your local environment:
   ```bash
   git clone https://github.com/leandric/oracle_chat.git
   cd oracle_chat
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the project, you will use Streamlit to launch the web application.

1. In the terminal, navigate to the project root directory and run the following command:
   ```bash
   streamlit run main.py
   ```

2. This will open a graphical interface in your browser where you can interact with the application. `Streamlit` typically runs at `http://localhost:8501`.

### Model Configuration

- **Supported Models**: The project allows you to choose between various AI models from Groq and OpenAI. For each model, you will need a valid **API Key**.
- **File Upload**: You can upload files (PDF, CSV, TXT), or input URLs for websites or YouTube videos to use the content in the AI model.

## How to Use

1. **Load Files**: Use the sidebar to select the file type you want to load (Website, YouTube, PDF, CSV, TXT). Depending on the type, you can either input a URL or upload a file.
   
2. **Select Model**: In the second tab on the sidebar, choose the model provider (Groq or OpenAI) and select the desired model. Enter your API key for the chosen provider.

3. **Initialize Oracle**: After uploading the file and selecting the model, click the "Initialize Oracle" button to start the assistant. The model will load the content, and you can begin asking questions through the chat interface.

4. **Clear History**: If you want to start a new conversation, you can click the "Clear Conversation History" button.

## Project Structure

- **`main.py`**: The main file that launches Streamlit and defines the interaction logic.
- **Loading Functions**: Functions like `load_pdf`, `load_csv`, and `load_youtube` are responsible for loading content from different file types.
- **`load_model`**: Sets up the AI model to interact with the provided content.
- **Interface**: The interface is built using `Streamlit`, utilizing the sidebar for file upload and model selection, and the main page for interacting with the Oracle.

## Technologies Used

- **Python 3.7+**
- **Streamlit** for the web interface.
- **Langchain** for AI model integration.
- **Groq** and **OpenAI** as AI model providers.
- **tempfile** for handling temporary file uploads.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
