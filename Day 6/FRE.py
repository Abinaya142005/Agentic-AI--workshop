import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.tools.retriever import create_retriever_tool

# STEP 1: Set Gemini API Key (paste directly)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDn9xRDMlWYcw1c2wyay5quz5jFyRPaZhU"  # 🔒 Replace with your Gemini API key

# STEP 2: Build vector store from PDF
def build_vectorstore():
    pdf_path = os.path.join("documents", "pitch_rejections.pdf")
    os.makedirs("documents", exist_ok=True)

    if not os.path.exists(pdf_path):
        st.error("📄 'pitch_rejections.pdf' not found in the 'documents/' folder.")
        return False

    with st.spinner("🔍 Indexing pitch rejection stories..."):
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            if not pages:
                st.error("⚠️ No readable content in the PDF.")
                return False

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(pages)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = FAISS.from_documents(chunks, embeddings)
            vectordb.save_local("faiss_index")

            st.success("✅ RAG knowledge base built successfully!")
            return True

        except Exception as e:
            st.error(f"❌ Error during indexing: {str(e)}")
            return False

# STEP 3: Load existing vector store
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"⚠️ Could not load vector store: {str(e)}")
        return None

# STEP 4: Calculate readiness score
def calculate_readiness_score(idea, team, mvp, traction):
    score = 0
    score += 20 if len(idea) > 50 and any(k in idea.lower() for k in ["problem", "solution", "unique", "market"]) else 10
    score += 20 if any(k in team.lower() for k in ["experience", "technical", "founder", "expertise"]) else 10
    score += 20 if any(k in mvp.lower() for k in ["prototype", "tested", "demo", "built"]) else 10
    score += 20 if any(k in traction.lower() for k in ["users", "revenue", "partnership", "waitlist"]) else 10
    return score

# STEP 5: Map score to funding stage
def map_to_funding_stage(score):
    if score < 40:
        return "Pre-seed (FFF or Angel only)", "Validate idea with interviews or build a basic prototype."
    elif score < 70:
        return "Pre-seed or Early Seed", "Develop and test MVP with 50+ users."
    else:
        return "Seed or Series A", "Scale traction to 1,000+ users or secure key partnerships."

# STEP 6: Create tool-calling agent
def get_agent():
    vectordb = load_vectorstore()
    if vectordb is None:
        return None

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever,
        name="startup_rejections_search",
        description="Searches investor rejection reasons and insights for startup pitches."
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a professional startup evaluator helping founders prepare for VC funding.
You will:
1. Give a VC readiness score out of 10 with reasoning.
2. Suggest a next milestone.
3. Support advice using retrieved investor rejection patterns.

Use this format:
**VC Readiness Score**: <score>/10  
**Explanation**: <your analysis>  
**Next Milestone**: <clear suggestion>  
"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=[retriever_tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)
    return executor

# STEP 7: Streamlit UI
def main():
    st.set_page_config(page_title="Fundraising Readiness Evaluator", page_icon="💰", layout="wide")
    st.title("🚀 Fundraising Readiness Evaluator")

    if not os.path.exists("faiss_index"):
        if not build_vectorstore():
            return

    st.subheader("📝 Enter Startup Details")
    with st.form("startup_form"):
        idea = st.text_area("📌 Startup Idea", height=100)
        team = st.text_area("👥 Team", height=100)
        mvp = st.text_area("🛠️ MVP Status", height=100)
        traction = st.text_area("📈 Traction", height=100)
        submitted = st.form_submit_button("💡 Evaluate", use_container_width=True)

    if submitted:
        if not all([idea, team, mvp, traction]):
            st.warning("⚠️ Please fill all fields.")
            return

        with st.spinner("🔄 Evaluating..."):
            score = calculate_readiness_score(idea, team, mvp, traction)
            funding_stage, milestone = map_to_funding_stage(score)
            agent = get_agent()
            if agent is None:
                st.error("⚠️ Agent could not be loaded.")
                return

            try:
                full_query = f"""Startup Evaluation Request:
Idea: {idea}
Team: {team}
MVP: {mvp}
Traction: {traction}"""

                result = agent.invoke({"input": full_query})

                st.markdown("### 📊 Evaluation Result")
                st.write(f"**Funding Stage**: {funding_stage}")
                st.write(f"**Structured Readiness Score**: {score}/100")
                st.markdown(result["output"])

                st.download_button(
                    label="📥 Download Evaluation",
                    data=f"Funding Stage: {funding_stage}\nScore: {score}/100\n\n{result['output']}",
                    file_name="evaluation.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"⚠️ Error during evaluation: {str(e)}")

# STEP 8: Run
if __name__ == "__main__":
    main()
