import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit page configuration
st.set_page_config(page_title="Fundraising Readiness Evaluator", layout="centered")

# Environment setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyAeFhw6AHJos-r7xq2xWTrk3jq2klChjDE"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)  # Increased temperature for more detailed responses

# Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF into vectorstore
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

# Initialize retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever() if vectorstore else None

# RAG function
def rag_query(query: str) -> str:
    if not retriever:
        return "Error: Investor feedback PDF could not be loaded. Please check the file."
    docs = retriever.get_relevant_documents(query)
    content = "\n".join([doc.page_content for doc in docs])
    prompt = f"Based on this data:\n{content}\nAnswer the question: {query}"
    return llm.invoke(prompt).content

# Tool for RAG (only for milestone_agent)
tools = [
    Tool(
        name="RAG_Query",
        func=rag_query,
        description="Use this to retrieve investor feedback and recommend next steps"
    )
]

# Initialize agents
# First three agents use CONVERSATIONAL_REACT_DESCRIPTION (no tools required)
assessment_agent = initialize_agent(
    tools=[],  # No tools needed
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose logging for debugging
    handle_parsing_errors=True  # Handle parsing errors
)

stage_mapper_agent = initialize_agent(
    tools=[],  # No tools needed
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose logging for debugging
    handle_parsing_errors=True  # Handle parsing errors
)

readiness_scorer_agent = initialize_agent(
    tools=[],  # No tools needed
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose logging for debugging
    handle_parsing_errors=True  # Handle parsing errors
)

# Milestone agent uses ZERO_SHOT_REACT_DESCRIPTION with RAG tool
milestone_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose logging for debugging
    handle_parsing_errors=True  # Handle parsing errors
)

# UI
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
                # 1. Startup Assessment Agent
                assessment_result = assessment_agent.invoke({
                    "input": f"""You are a Startup Assessment Agent. Evaluate the founder input for:
                    - Idea Clarity: Is the startup idea clear and well-defined? Provide a detailed explanation.
                    - Team Composition: Does the team have relevant skills and experience? Include specific observations.
                    - MVP Status: Is there a minimum viable product? Describe its development stage and functionality.
                    - Traction: What evidence of traction (real or simulated) exists? Provide a detailed analysis.
                    Provide a structured response with bullet points for each criterion. Do not summarize; include a detailed explanation for each point.
                    Input:
                    {founder_input}""",
                    "chat_history": []  # Empty chat history
                })["output"]

                # 2. Funding Stage Mapper Agent
                stage_result = stage_mapper_agent.invoke({
                    "input": f"""You are a Funding Stage Mapping Agent. Determine:
                    - Startup Funding Stage: Identify the appropriate funding stage (e.g., Pre-seed, Seed, Series A) based on the startup's progress and needs. Provide a detailed explanation of the reasoning.
                    - Matching Investor Type: Recommend investor types (e.g., FFF, Angel, VC) that align with the stage and startup profile. Explain why these investors are suitable, including specific investor characteristics.
                    Provide a structured response with bullet points for each item. Do not summarize; include a detailed explanation for each point.
                    Input:
                    {founder_input}""",
                    "chat_history": []  # Empty chat history
                })["output"]

                # 3. Readiness Scoring Agent
                score_result = readiness_scorer_agent.invoke({
                    "input": f"""You are a Readiness Scoring Agent. Evaluate the founder input and provide:
                    - Product-Market Fit: Assign a score out of 100 based on how well the startup addresses a market need. Provide a detailed explanation, including evidence from the input.
                    - Growth Signals: Assign a score out of 100 based on evidence of growth potential (e.g., user adoption, revenue, partnerships). Include a detailed analysis.
                    - Founder-Market Alignment: Assign a score out of 100 based on how well the founders' skills and experience align with the market. Justify with specific details.
                    Provide a structured response with bullet points for each criterion, including the score and a detailed explanation. Do not summarize; address each criterion fully.
                    Input:
                    {founder_input}""",
                    "chat_history": []  # Empty chat history
                })["output"]

                # 4. Milestone Re SEGUEcommender Agent using RAG
                milestone_result = milestone_agent.invoke({
                    "input": f"""You are a Milestone Recommendation Agent. Use the RAG_Query tool to retrieve investor feedback and suggest next steps for the startup. Provide a structured response with bullet points listing specific milestones or actions the startup should take, along with detailed explanations based on the feedback.
                    Input:
                    {founder_input}"""
                })["output"]

                st.success("✅ Evaluation Complete!")
                st.subheader("🔍 Results")

                st.markdown(f"*1. 📊 Startup Assessment:*\n\n{assessment_result}", unsafe_allow_html=True)
                st.markdown(f"*2. 🧭 Funding Stage Mapping:*\n\n{stage_result}", unsafe_allow_html=True)
                st.markdown(f"*3. 🧠 Readiness Score:*\n\n{score_result}", unsafe_allow_html=True)
                st.markdown(f"*4. 📚 Milestone Recommendations (RAG):*\n\n{milestone_result}", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")