import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai



# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is not None:
    genai.api_key = api_key
else:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Specify the local path for PDF folder
LOCAL_PDF_FOLDER_PATH = "/path/to/your/pdf/folder"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Safely handle FAISS loading
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response.get("output_text", "No response generated."))
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")

def load_pdfs_from_folder(folder_path):
    pdf_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            pdf_files.append(file_path)
    return pdf_files

def main():
    st.set_page_config("Chat PDF")
    st.header("Digging Deep into the Earth")

    user_question = st.text_input("Ask a Question")

    # Load PDFs from local folder
    local_pdfs = load_pdfs_from_folder("PDFs")

    

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        # File uploader for additional PDFs
        uploaded_pdfs = st.file_uploader("Upload additional PDF Files", type="pdf", accept_multiple_files=True)

        if uploaded_pdfs:
            pdf_docs = local_pdfs + [pdf for pdf in uploaded_pdfs]
        else:
            pdf_docs = local_pdfs
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
