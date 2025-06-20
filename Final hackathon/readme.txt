Project Description

This project implements a **Multi-Agent AI System** to automate project analysis, purpose generation, feedback extraction, process planning, and diagram generation for student learning applications.

It uses **Streamlit for the user interface**, **LangChain Agents for orchestration**, and the **Google Gemini API** for natural language processing tasks.

Features

* Automatic project context summarization.
* Purpose statement generation.
* Market feedback analysis using PDF documents.
* 5-stage stepwise process planning.
* MermaidJS flowchart generation.
* Data storage using SQLite.


 🛠️ **Technology Stack**

* Python
* Streamlit
* LangChain
* Google Generative AI (Gemini)
* FAISS (Vector Database)
* SQLite (Database)



**Installation**

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Install required packages:

   ```bash
   pip install streamlit langchain langchain-google-genai langchain-core langchain-community faiss-cpu pypdf
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

 **API Key Setup**

* Update the following line in `app.py` with your **Google Gemini API Key:**

  ```python
  api_key = "YOUR_API_KEY"
  ```

---
 **File Requirements**

* Ensure the following PDF file is available at this path:

  ```text
  C:/Users/ABINAYA.S/Desktop/hackathon final/Market_Feedback_Sample.pdf
  ```

---

### 🧩 **Agent Overview**

| Agent   | Role                                                |
| ------- | --------------------------------------------------- |
| Agent 1 | Extracts concise project context summary.           |
| Agent 2 | Generates purpose statements based on the context.  |
| Agent 3 | Analyzes feedback PDF to identify feature gaps.     |
| Agent 4 | Creates a detailed 5-stage process plan.            |
| Agent 5 | Generates MermaidJS diagrams from the process plan. |

---

 **Database Details**

* **Database Name:** `database.db`
* **Tables:**

  * `projects`: Stores project title, metadata, and feedback.
  * `purpose_statements`: Stores purpose statements linked to projects.
  * `process_flows`: Stores process plans and diagrams linked to projects.

---

 **How to Use**

1. Open the Streamlit app in your browser.
2. Enter:

   * Project Title
   * Project Metadata
   * Project Feedback
3. Click **Run All Agents**.
4. View outputs in the following tabs:

   * Context Summary
   * Purpose Statement
   * Feedback Analysis
   * Process Plan
   * Process Diagram (MermaidJS)


