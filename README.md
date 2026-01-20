# Chat with PDF 📚

This is a Streamlit web application that allows you to chat with your PDF documents. Upload one or more PDFs, and the application will process them, enabling you to ask questions and get answers based on the content of the documents.

## Features

-   **Interactive Chat Interface:** A modern and intuitive chat UI to interact with your documents.
-   **Multiple PDF Upload:** Upload one or more PDF files at once.
-   **Secure API Key Management:** Uses a `.env` file to securely manage your OpenAI API key.
-   **Powered by LangChain and OpenAI:** Leverages the power of large language models for question-answering.
-   **Easy to Use:** Simple sidebar configuration for uploading and processing documents.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.8+
-   `pip` for package management

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Bootcamp-updates/Chat_with_pdf
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

Create a `.env` file in the root of the project directory (`Chat_with_pdf`) and add your OpenAI API key:

```
OPENAI_API_KEY="your-openai-api-key-here"
```

## Usage

Once the setup is complete, you can run the Streamlit application.

1.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

2.  **Open in Browser:**
    The application will open in your default web browser.

3.  **Upload and Process:**
    -   Use the sidebar to upload one or more PDF files.
    -   Click the "Process" button to let the application read and index your documents.

4.  **Start Chatting:**
    -   Once processing is complete, you can start asking questions about your documents in the chat input box.
