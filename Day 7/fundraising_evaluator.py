import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent

st.set_page_config(page_title="Fundraising Readiness Evaluator", layout="centered")

os.environ["GOOGLE_API_KEY"] = "AIzaSyAeFhw6AHJos-r7xq2xWTrk3jq2klChjDE"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -------------------- RAG SETUP --------------------
@st.cache_resource
def load_vectorstore():
    try:
        loader = PyPDFLoader("investor_feedback.pdf")
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever() if vectorstore else None

def rag_query(query: str) -> str:
    if not retriever:
        return "Document data not available. Please check PDF loading."
    docs = retriever.get_relevant_documents(query)
    content = "\n".join([doc.page_content for doc in docs])
    return llm.invoke(f"Based on this data:\n{content}\nAnswer the question: {query}").content


def startup_assessment(inputs: dict) -> str:
    prompt = f"""Evaluate the founder input for:
    - Idea Clarity
    - Team Composition
    - MVP Status
    - Traction (real/simulated)

    Input: {inputs['founder_input']}"""
    return llm.invoke(prompt).content

def funding_stage_mapper(inputs: dict) -> str:
    prompt = f"""Given the startup description, determine:
    - Funding stage (Pre-seed, Seed, etc.)
    - Matching investor type (FFF, Angel, VC)

    Input: {inputs['founder_input']}"""
    return llm.invoke(prompt).content

def readiness_scorer(inputs: dict) -> str:
    prompt = f"""Evaluate and score:
    - Product-market fit
    - Growth signals
    - Founder-market alignment

    Give a readiness score out of 100.

    Input: {inputs['founder_input']}"""
    return llm.invoke(prompt).content

def milestone_recommender(inputs: dict) -> str:
    return rag_query(f"What should the startup do next based on this input: {inputs['founder_input']}")

# -------------------- STREAMLIT UI --------------------
st.title("🚀 Fundraising Readiness Evaluator (Multi-Agent AI)")

st.markdown("This tool analyzes your startup input using 4 intelligent agents:")
st.markdown("- 📊 **Startup Assessment Agent**")
st.markdown("- 🧭 **Funding Stage Mapper**")
st.markdown("- 🧠 **Readiness Scorer**")
st.markdown("- 📚 **Milestone Recommender (RAG)**")

founder_input = st.text_area("📝 Describe your startup", height=250, placeholder="Your idea, team, MVP status, traction, etc.")

if st.button("Run Evaluation"):
    if not founder_input.strip():
        st.warning("Please provide a description of your startup.")
    else:
        with st.spinner("Running all 4 agents..."):
            inputs = {"founder_input": founder_input}

            assessment = startup_assessment(inputs)
            stage = funding_stage_mapper(inputs)
            score = readiness_scorer(inputs)
            milestones = milestone_recommender(inputs)

        st.success("✅ Evaluation Complete!")
        st.subheader("🔍 Results")

        st.markdown(f"**1. 📊 Startup Assessment:**\n\n{assessment}", unsafe_allow_html=True)
        st.markdown(f"**2. 🧭 Funding Stage Mapping:**\n\n{stage}", unsafe_allow_html=True)
        st.markdown(f"**3. 🧠 Readiness Score:**\n\n{score}", unsafe_allow_html=True)
        st.markdown(f"**4. 📚 Milestone Recommendations (RAG):**\n\n{milestones}", unsafe_allow_html=True)
