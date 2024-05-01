import streamlit as st
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os

api_key1 = st.secrets["google_api_key"]
os.environ['GOOGLE_API_KEY'] = api_key1

llm = ChatGoogleGenerativeAI(model="gemini-pro")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)

# Define prompts
chunks_prompt="""
You are a reviewer tasked with evaluating a research paper for acceptance or rejection. 
The paper consists of several sections, including Introduction, Methodology, Results, Discussion, and Conclusion. 
Your task is to provide detailed feedback for each section and suggest areas of improvement. 
Each section should contain 5 points for improvement scope and feedback.
Additionally, assign a score ranging from 1 to 5 for each section based on its quality and coherence
Paper:{text}
Review Notes:
"""

map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

final_combine_prompt="""
Provide a final feedback of all sections and the improvement required in each sections.
Also provide a final score range(1-10) and tell whether to accept,reject,accept with revision in a high quality journal.
The ouput you are giving give it in this format:
Abstract:
Score:
Improvement required:
1.
2.
3.
4.
5.
Introduction and so on.
At last Final Score:
Recommendation:
Note- Each section should contain 5 points for improvement scope and feedback. (dont give** anywhere)
Paper: {text}
"""


final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)

# Load summarization chain
summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template,
                                     combine_prompt=final_combine_prompt_template, verbose=False)


def extract_text_from_pdf(file):
    text = ''
    pdfreader = PdfReader(file)
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def generate_summary(text):
    # Split text into chunks
    chunks = text_splitter.create_documents([text])

    # Generate summary
    output = summary_chain.run(chunks)

    return output


def main():
    st.set_page_config("ResearchRadar 📄")
    st.text("Designed by Subhayu Dutta 2024")
    navbar = """
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #CDFADB;
            padding: 20px;
            margin-bottom: 20px;
        }
        .navbar-title {
            font-size: 24px;
            font-weight: bold;
            margin-right: 10px;
        }
        .navbar-logo {
            height: 40px;
        }
    </style>
    
    <div class="navbar">
        <div class="navbar-title">ResearchRadar</div>
        <img class="navbar-logo" src="https://i0.wp.com/thepublicationplan.com/wp-content/uploads/2021/08/Detectives-investigating-paper-mills.jpg?fit=585%2C596&ssl=1" alt="Logo">
    </div>
    """
    st.markdown(navbar, unsafe_allow_html=True)
    s="This a platform dedicated to evaluating research papers, providing scores, and offering recommendations based on their quality and relevance."
    st.write(s)

    # s="This a platform dedicated to evaluating research papers, providing scores, and offering recommendations based on their quality and relevance."
    # st.sidebar.markdown(f'<div style="color:black;background-color:#F6F5F2; padding:10px; border-radius:5px;">{s}</div>',
    #                         unsafe_allow_html=True)
    # Sidebar layout
    st.sidebar.title("Upload Research Paper in PDF format:")
    uploaded_file = st.sidebar.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.subheader("Feedback Provided After Review")
        with st.container():
            summary = generate_summary(text)
            st.markdown(f'<div style="color:black;background-color:#F6F5F2; padding:10px; border-radius:5px;">{summary}</div>',
                            unsafe_allow_html=True)



if __name__ == "__main__":
    main()
