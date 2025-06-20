import os
import sqlite3
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool

# ✅ API Key
api_key = "AIzaSyBzEXYCUyp2jBjhTZgqz29cq6Sq8D7UyGQ"

# --------------------- Streamlit Config ---------------------
st.set_page_config(page_title="Agentic AI Multi-Agent System", layout="wide")
st.title("🌟 Agentic AI Multi-Agent System")

# --------------------- Database Functions ---------------------
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        metadata TEXT,
                        feedback TEXT
                      )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS purpose_statements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        statement TEXT
                      )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS process_flows (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        flow_plan TEXT,
                        diagram TEXT
                      )''')

    conn.commit()
    conn.close()

def insert_project(title, metadata, feedback):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO projects (title, metadata, feedback) VALUES (?, ?, ?)', (title, metadata, feedback))
    conn.commit()
    project_id = cursor.lastrowid
    conn.close()
    return project_id

def insert_purpose(project_id, statement):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO purpose_statements (project_id, statement) VALUES (?, ?)', (project_id, statement))
    conn.commit()
    conn.close()

def insert_process_flow(project_id, flow_plan, diagram):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO process_flows (project_id, flow_plan, diagram) VALUES (?, ?, ?)', (project_id, flow_plan, diagram))
    conn.commit()
    conn.close()

# --------------------- PDF Loader ---------------------
def load_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    return pages

# --------------------- Agent Tools ---------------------
def context_reader_tool(input_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, convert_system_message_to_human=True)
    response = llm.invoke([HumanMessage(content=input_text)])
    return response.content.strip()

def purpose_generator_tool(input_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, convert_system_message_to_human=True)
    response = llm.invoke([HumanMessage(content=input_text)])
    return response.content.strip()

def feedback_analyzer_tool(input_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, convert_system_message_to_human=True)
    docs = load_pdf('C:/Users/ABINAYA.S/Desktop/hackathon final/Market_Feedback_Sample.pdf')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    query = input_text
    relevant_docs = retriever.get_relevant_documents(query)
    combined_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    response = llm.invoke([HumanMessage(content=f"""
        Based on the following documents:
        {combined_context}

        List 3 common complaints or feature gaps related to student learning apps. Be specific.

        Do not add any introduction or explanation. Provide the list only.
    """)])
    return response.content.strip()

def process_planner_tool(input_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, convert_system_message_to_human=True)
    response = llm.invoke([HumanMessage(content=input_text)])
    return response.content.strip()

def diagram_composer_tool(input_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, convert_system_message_to_human=True)
    response = llm.invoke([HumanMessage(content=input_text)])
    return response.content.strip()

# --------------------- Agent Initialization ---------------------
def initialize_agents():
    agent1 = initialize_agent(
        tools=[Tool.from_function(context_reader_tool, name="ContextReader", description="Reads project context and generates a concise summary.")],
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    agent2 = initialize_agent(
        tools=[Tool.from_function(purpose_generator_tool, name="PurposeGenerator", description="Generates purpose statement for student learning app.")],
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    agent3 = initialize_agent(
        tools=[Tool.from_function(feedback_analyzer_tool, name="FeedbackAnalyzer", description="Analyzes PDF to find feature gaps and complaints.")],
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    agent4 = initialize_agent(
        tools=[Tool.from_function(process_planner_tool, name="ProcessPlanner", description="Creates a detailed 5-stage process plan.")],
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    agent5 = initialize_agent(
        tools=[Tool.from_function(diagram_composer_tool, name="DiagramComposer", description="Creates MermaidJS diagrams from process plans.")],
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    return agent1, agent2, agent3, agent4, agent5

# --------------------- Streamlit Input Section ---------------------
project_title = st.text_input("Enter Project Title")
project_metadata = st.text_area("Enter Project Metadata")
project_feedback = st.text_area("Enter Project Feedback")

if st.button("Run All Agents"):
    if project_title and project_metadata and project_feedback:
        with st.spinner("Running agents... Please wait."):
            init_db()
            project_id = insert_project(project_title, project_metadata, project_feedback)

            agent1, agent2, agent3, agent4, agent5 = initialize_agents()

            context_prompt = f"""
                You are analyzing the {project_title} project.

                Extract a project context summary in EXACTLY 5 lines.
                Focus on:
                - Key student user needs
                - Learning opportunity areas
                - Project goals related to student learning
                - Relevant student pain points
                - Implementation concerns specific to educational apps

                Do not add any introduction or explanation. Provide ONLY the 5-line summary.

                Input Details:
                metadata: {project_metadata}
                feedback: {project_feedback}
            """

            context_summary = agent1.run(context_prompt)

            purpose_prompt = f"""
                Based on this project context:
                {context_summary}

                Generate a purpose statement in this format:
                'My product/app will help [target audience] with [user problem] by [solution].'

                Replace all placeholders. Be specific. Do not add any extra explanation.
            """

            purpose_statement = agent2.run(purpose_prompt)

            feedback_prompt = f"What are the common complaints or feature gaps in apps similar to: {context_summary}"
            refined_purpose = agent3.run(feedback_prompt)

            process_prompt = f"""
                You are a project planning expert for student learning apps.

                Based on this project:
                {refined_purpose}

                Create a 5-stage stepwise process plan focusing on these features:
                - Progress tracking
                - Personalized learning paths
                - Timely reminders
                - Student engagement
                - Learning analytics

                For each stage, provide:
                - Stage Name
                - Milestones
                - Checkpoints
                - Inputs and Outputs
                - Dependencies
                - Contingency plans

                Ensure the process is directly relevant to the user-provided project context.
                Do not add any extra summary. Provide the process plan only.
            """

            process_plan = agent4.run(process_prompt)

            diagram_prompt = f"""
                Convert the following process plan into a MermaidJS flowchart.

                Each stage should:
                - Be labeled with its student learning function.
                - Include the expected student outcome.

                Provide ONLY the MermaidJS diagram code. Do not add explanations or summary.

                Process Plan:
                {process_plan}
            """

            diagram = agent5.run(diagram_prompt)

            insert_purpose(project_id, purpose_statement)
            insert_process_flow(project_id, process_plan, diagram)

        st.success("✅ All agents executed successfully!")

        # Display outputs in separate tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Context Summary", "🎯 Purpose Statement", "💬 Feedback Analysis", "🛠️ Process Plan", "📈 Process Diagram"
        ])

        with tab1:
            st.subheader("🔍 Context Summary")
            st.write(context_summary)

        with tab2:
            st.subheader("🎯 Purpose Statement")
            st.write(purpose_statement)

        with tab3:
            st.subheader("💬 Feedback Analysis")
            st.write(refined_purpose)

        with tab4:
            st.subheader("🛠️ Process Plan")
            st.write(process_plan)

        with tab5:
            st.subheader("📈 Process Diagram (MermaidJS)")
            st.code(diagram, language="text")

    else:
        st.warning("⚠️ Please fill all the input fields before running the agents.")
