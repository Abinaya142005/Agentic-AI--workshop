import streamlit as st
import PyPDF2
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ========================= Gemini Setup =========================
genai.configure(api_key="AIzaSyDvmXdoYK0Nu3OlOV8q_KYpZXgodSWir5E")  # 🔁 Replace this with your actual API key

# ==================== Custom Gemini LangChain Wrapper ====================
class GeminiLLM(LLM, BaseModel):
    model_name: str = "gemini-2.0-flash"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self) -> str:
        return "gemini"

# ==================== Prompt Functions ====================
def summarize_content(text):
    template = """
    Summarize the following study material into clear bullet points:

    {study_material}

    Summary:
    """
    prompt = PromptTemplate(input_variables=["study_material"], template=template)
    llm = GeminiLLM()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(study_material=text)

def generate_mcqs(text):
    template = """
    Create 3 multiple-choice questions based on the following study material. Each question should have 4 options and the correct answer clearly marked.

    Study Material:
    {study_material}

    Output:
    """
    prompt = PromptTemplate(input_variables=["study_material"], template=template)
    llm = GeminiLLM()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(study_material=text)

# ==================== PDF Text Extraction ====================
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# ==================== Streamlit UI ====================
st.title("📘 Study Assistant with Gemini")
st.write("Upload a PDF to get a summary and auto-generated MCQs.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and analyzing the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Summary
        st.subheader("📋 Summary")
        summary = summarize_content(pdf_text)
        st.write(summary)

        # MCQs
        st.subheader("❓ Multiple Choice Questions")
        mcqs = generate_mcqs(pdf_text)
        st.markdown(mcqs.replace("**", ""), unsafe_allow_html=True)
