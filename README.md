# AI HR Recruitment Assistant

IBM Internship Use Case: **AI HR Recruitment Assistant**

## What this project does

1. HR enters a Job Description.
2. HR uploads a candidate resume.
3. The app extracts resume text.
4. RAG retrieves relevant HR/recruitment knowledge.
5. An AI agent uses tools for skill extraction and match scoring.
6. The LLM produces:
   - candidate skills
   - matching skills
   - missing skills
   - match score
   - strengths/weaknesses
   - interview questions
   - recommendation
7. Streamlit displays the final recruitment report.

## Architecture

Resume + Job Description
        |
        v
  Text Extraction
        |
        +----> Chroma Vector DB <---- HR knowledge base
        |             |
        +-------------+
                      v
                Recruitment Agent
                      |
             +--------+--------+
             |                 |
        Skill Match Tool   Knowledge Search
             |                 |
             +--------+--------+
                      v
                     LLM
                      |
                      v
              Recruitment Report

## Setup

### 1. Create a virtual environment

Windows:

    python -m venv venv
    venv\Scripts\activate

### 2. Install packages

    pip install -r requirements.txt

### 3. Create .env

Copy `.env.example` to `.env` and add your OpenAI API key.

### 4. Build the RAG database

    python ingest.py

### 5. Run the app

    streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

## Demo data

- `data/resumes/sample_resume.txt`
- `data/job_descriptions/sample_job_description.txt`
- `data/knowledge_base/recruitment_policy.txt`

## Important project concepts

- **RAG:** Chroma vector database + embeddings + retriever
- **Agent:** LangChain agent
- **Tools:** skill extraction, skill matching, knowledge search
- **LLM:** OpenAI chat model
- **UI:** Streamlit

## Important note

This is a prototype for an internship/demo. It should support human HR review rather than making final employment decisions automatically.
