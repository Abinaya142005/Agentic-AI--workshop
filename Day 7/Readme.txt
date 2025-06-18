# 🚀 Fundraising Readiness Evaluator (Multi-Agent AI System)

A multi-agent intelligent evaluator built with LangChain, Google Gemini, FAISS, and Streamlit. It helps early-stage startups assess their readiness for fundraising and receive structured, personalized feedback — powered by RAG (Retrieval-Augmented Generation).

---

## 📌 Overview

Many startups pitch to investors too early — without validating key components like MVP, traction, or funding stage fit. This tool uses **4 AI agents** to help founders self-evaluate and identify next milestones:

### 🤖 Agents:

1. **📊 Startup Assessment Agent**
   Evaluates:

   * Idea Clarity
   * Team Composition
   * MVP Status
   * Traction

2. **🧭 Funding Stage Mapping Agent**
   Maps:

   * Startup maturity to funding stages (e.g., Pre-seed, Seed)
   * Suitable investor types (e.g., FFF, Angel, VC)

3. **🧠 Readiness Scoring Agent**
   Scores:

   * Product-Market Fit
   * Growth Signals
   * Founder-Market Alignment

4. **📚 Milestone Recommender Agent (RAG)**
   Uses investor feedback PDF (via RAG) to recommend next actionable milestones.

---

## 🛠️ Tech Stack

| Component           | Technology                                       |
| ------------------- | ------------------------------------------------ |
| Frontend            | Streamlit                                        |
| Backend             | Python                                           |
| Agents              | LangChain                                        |
| LLM                 | Google Gemini Pro (via `langchain_google_genai`) |
| Vector DB           | FAISS                                            |
| Embeddings          | Sentence Transformers (HuggingFace)              |
| Document Processing | PyPDFLoader                                      |

---

## 🧩 Features

* ✅ Multi-agent coordination for deep startup evaluation
* 🧠 LLM reasoning without tools (for 3 agents)
* 🔎 Retrieval-Augmented Generation (RAG) for milestone recommendations
* 📄 Upload and analyze investor feedback (PDF)
* 🖼️ Interactive UI with Streamlit

---

## 📂 Folder Structure

```
Hack-02/
├── pro/
│   └── fundraising_evaluator.py
├── investor_feedback.pdf
├── README.md
└── requirements.txt
```

---

## 🚀 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/fundraising-readiness-evaluator.git
cd fundraising-readiness-evaluator/pro
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r ../requirements.txt
```

### 4. Add your PDF

Ensure `investor_feedback.pdf` is present in the project root. This file is used by the RAG-enabled agent.

### 5. Add your API key

Edit this line in `fundraising_evaluator.py`:

```python
os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"
```

### 6. Run the app

```bash
streamlit run fundraising_evaluator.py
```

---

## 🧠 Example Input

> "We are building an AI-powered marketplace that connects farmers directly with retailers. Our team includes an ex-AgriTech PM, a data scientist, and two engineers. We have a WhatsApp-based MVP with 150 early users. No revenue yet."

---

## 🧾 Output Example

* 📊 Structured assessment of idea, team, MVP, and traction
* 🧭 Funding stage suggestion: *Pre-Seed → Angel Investors*
* 🧠 Scores out of 100 for PMF, growth potential, and founder-market alignment
* 📚 Milestone plan based on investor expectations (RAG output)

---

## ✅ Dependencies

> Create `requirements.txt` if not already created:

```
streamlit
langchain
langchain-community
langchain-google-genai
sentence-transformers
faiss-cpu
```

---

## 📜 License

MIT License. Use freely with attribution.

