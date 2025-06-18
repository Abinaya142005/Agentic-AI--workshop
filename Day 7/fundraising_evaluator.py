import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings 

# Streamlit page configuration
st.set_page_config(page_title="Fundraising Readiness Evaluator", layout="centered")

# Environment setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyAeFhw6AHJos-r7xq2xWTrk3jq2klChjDE"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ✅ Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cache vectorstore loading
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

# Initialize vectorstore and retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever() if vectorstore else None

# RAG tool for milestone recommendations
def rag_query(query: str) -> str:
    if not retriever:
        return "Document data not available. Please check PDF loading."
    docs = retriever.get_relevant_documents(query)
    content = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        "Based on this data:\n{content}\nAnswer the question: {query}"
    )
    return llm.invoke(prompt.format(content=content, query=query)).content

# Define tools
tools = [
    Tool(
        name="RAG_Query",
        func=rag_query,
        description="Retrieve and generate answers from investor feedback documents"
    )
]

# Agent prompt templates with {agent_scratchpad}
assessment_prompt = ChatPromptTemplate.from_template(
    """You are a Startup Assessment Agent. Evaluate the founder input for:
    - Idea Clarity
    - Team Composition
    - MVP Status
    - Traction (real/simulated)
    
    Input: {founder_input}
    
    Provide a detailed assessment of each aspect.
    
    {agent_scratchpad}"""
)

stage_mapper_prompt = ChatPromptTemplate.from_template(
    """You are a Funding Stage Mapping Agent. Given the startup description, determine:
    - Appropriate funding stage (Pre-seed, Seed, Series A, etc.)
    - Matching investor type (FFF, Angel, VC)
    
    Input: {founder_input}
    
    Provide a clear mapping with justifications.
    
    {agent_scratchpad}"""
)

readiness_scorer_prompt = ChatPromptTemplate.from_template(
    """You are a Readiness Scoring Agent. Evaluate and score:
    - Product-market fit
    - Growth signals
    - Founder-market alignment
    
    Give a readiness score out of 100 and explain the scoring.
    
    Input: {founder_input}
    
    Thoughts and Actions so far:
{agent_scratchpad}"""
)

milestone_prompt = ChatPromptTemplate.from_template(
    """You are a Milestone Recommendation Agent. Use the RAG_Query tool to recommend next steps for the startup based on investor feedback.
    
    Input: {founder_input}
    
    {agent_scratchpad}"""
)

# Create agents
assessment_agent = create_tool_calling_agent(llm, [], assessment_prompt)
stage_mapper_agent = create_tool_calling_agent(llm, [], stage_mapper_prompt)
readiness_scorer_agent = create_tool_calling_agent(llm, [], readiness_scorer_prompt)
milestone_agent = create_tool_calling_agent(llm, tools, milestone_prompt)

# Create agent executors
assessment_executor = AgentExecutor(agent=assessment_agent, tools=[], verbose=False)
stage_mapper_executor = AgentExecutor(agent=stage_mapper_agent, tools=[], verbose=False)
readiness_scorer_executor = AgentExecutor(agent=readiness_scorer_agent, tools=[], verbose=False)
milestone_executor = AgentExecutor(agent=milestone_agent, tools=tools, verbose=False)

# Streamlit UI
st.title("🚀 Fundraising Readiness Evaluator (Multi-Agent AI)")

st.markdown("This tool analyzes your startup input using 4 intelligent agents:")
st.markdown("- 📊 *Startup Assessment Agent*: Evaluates idea, team, MVP, and traction")
st.markdown("- 🧭 *Funding Stage Mapper*: Maps to appropriate funding stage and investor types")
st.markdown("- 🧠 *Readiness Scorer*: Scores product-market fit, growth, and alignment")
st.markdown("- 📚 *Milestone Recommender*: Provides next steps using RAG")

founder_input = st.text_area(
    "📝 Describe your startup",
    height=250,
    placeholder="Your idea, team, MVP status, traction"
)

if st.button("Run Evaluation"):
    if not founder_input.strip():
        st.warning("Please provide a description of your startup.")
    else:
        with st.spinner("Running all 4 agents..."):
            try:
                inputs = {"founder_input": founder_input}

                # Execute agents
                assessment_result = assessment_executor.invoke(inputs)["output"]
                stage_result = stage_mapper_executor.invoke(inputs)["output"]
                score_result = readiness_scorer_executor.invoke(inputs)["output"]
                milestone_result = milestone_executor.invoke({
                    "founder_input": f"What should the startup do next based on this input: {founder_input}"
                })["output"]

                st.success("✅ Evaluation Complete!")
                st.subheader("🔍 Results")

                st.markdown(f"*1. 📊 Startup Assessment:*\n\n{assessment_result}", unsafe_allow_html=True)
                st.markdown(f"*2. 🧭 Funding Stage Mapping:*\n\n{stage_result}", unsafe_allow_html=True)
                st.markdown(f"*3. 🧠 Readiness Score:*\n\n{score_result}", unsafe_allow_html=True)
                st.markdown(f"*4. 📚 Milestone Recommendations (RAG):*\n\n{milestone_result}", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")
