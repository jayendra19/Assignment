import os
from langchain.vectorstores import Chroma#USED FOR STORING VECTOR EMBEDINGS
from langchain.text_splitter import RecursiveCharacterTextSplitter#THIS IS USED FOR USING FOR SPLIT THE TEXTS 
from langchain.chains import RetrievalQA#The RetrievalQA method is designed for large numbers of documents that exceed the context window of the LLM. It uses a retriever to select the most relevant documents for question answering, which improves speed and accuracy.
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain#this is design for small number of document It uses all documents for question answering, which might be inefficient and redundant.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import PyPDF2 as pdf
from langchain.prompts import PromptTemplate
import docx
import textract
from PyPDF2 import PdfReader



def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory_or_file):
    combined_text = ""

    if os.path.isdir(directory_or_file):
        # Process all files in the directory
        for filename in os.listdir(directory_or_file):
            file_path = os.path.join(directory_or_file, filename)
            if filename.endswith(".pdf"):
                combined_text += read_pdf(file_path)
            elif filename.endswith(".docx"):
                combined_text += read_word(file_path)
            elif filename.endswith(".txt"):
                combined_text += read_txt(file_path)
    else:
        # Treat the input as a single file
        if directory_or_file.endswith(".pdf"):
            combined_text += read_pdf(directory_or_file)
        elif directory_or_file.endswith(".docx"):
            combined_text += read_word(directory_or_file)
        elif directory_or_file.endswith(".txt"):
            combined_text += read_txt(directory_or_file)
        else:
            print(f"Unsupported file format: {directory_or_file}")

    return combined_text

    

def answer_user_question(question,path):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")#we can also use this for embdedings this is for goodle if u want to use openai embedding 

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,convert_system_message_to_human=True)
    
    local_file_path=path
    loader = PyPDFLoader(local_file_path)
    documents = loader.load_and_split()
    #print(local_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)#the length of the list of texts that are returned by the tool. It means that the tool has split the large text into 12 smaller chunks,
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma(embedding_function=embeddings)
    vectordb.add_documents(texts)
    #retriever = vectordb.as_retriever(search_kwargs={"k": 2})#using this getting relevant document for query
    similar=vectordb.similarity_search(question)
    prompt_temp="""Answer the question as detailed as possible from the provided context,make sure to provide all the details ,if answer is not in the 
    provided context just say ,"answer is not available in the provided context",don't provide the wrong answer\n\n\
    Context:\n {context}?\n
    question:\n{question}\n

    Answer:
    """

    prompts=PromptTemplate(template=prompt_temp,input_variables=["context","question"])
    chain=load_qa_chain(llm,chain_type="stuff",prompt=prompts)
    response = chain({"input_documents": similar, "question": user_question}, return_only_outputs=True)

    response_text = response["output_text"]
    
    return response_text


# Example usage
resume_path = r"C:\Users\jayen\Project\SJS Transcript Call.pdf"
user_question = "give me the summary of my pdf"
answer = answer_user_question(user_question, resume_path)
print(answer)




