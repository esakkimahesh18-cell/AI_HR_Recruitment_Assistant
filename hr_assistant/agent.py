import os

from langchain_ollama import ChatOllama

from .tools import (
    extract_skills,
    calculate_match_score,
    search_hr_knowledge,
)


def create_recruitment_agent():
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    return ChatOllama(
        model=model_name,
        temperature=0,
    )


def analyze_candidate(job_description: str, resume_text: str) -> str:
    # 1. Extract skills from Job Description
    required_skills = extract_skills.invoke({
        "text": job_description
    })

    # 2. Extract skills from Resume
    candidate_skills = extract_skills.invoke({
        "text": resume_text
    })

    # 3. Calculate deterministic match score
    match_result = calculate_match_score.invoke({
        "required_skills": required_skills,
        "candidate_skills": candidate_skills,
    })

    # 4. Retrieve HR knowledge using RAG
    hr_knowledge = search_hr_knowledge.invoke({
        "query": "recruitment policy interview guidelines technical roles"
    })

    # 5. Ask the local LLM to generate the final report
    prompt = f"""
You are an AI HR Recruitment Assistant.

Analyze the candidate strictly using the information provided below.

IMPORTANT RULES:
1. Do not invent candidate skills.
2. Do not invent job requirements.
3. Do not create individual skill percentages.
4. Use the Match Score provided below exactly.
5. Missing skills must be described as development gaps.
6. Interview questions must relate to the job description and candidate experience.
7. Include at least one technical problem-solving question.
8. Do not use protected personal characteristics.
9. The final hiring decision must be made by a human recruiter.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

REQUIRED SKILLS IDENTIFIED:
{required_skills}

CANDIDATE SKILLS IDENTIFIED:
{candidate_skills}

DETERMINISTIC MATCH RESULT:
{match_result}

HR KNOWLEDGE FROM RAG:
{hr_knowledge}

Generate a concise recruitment report with exactly these sections:

## Candidate Summary
Briefly summarize the candidate.

## Skills Match
List the matched required skills.

## Missing Skills
List only the missing required skills.

## Match Score
Use the deterministic Match Score provided above.
Do not calculate another score.

## Recommendation
Choose one:
- Strong Match
- Moderate Match
- Low Match

Explain the recommendation briefly using job-related evidence.

## Interview Questions
Generate 5 relevant technical interview questions.
Include:
- questions about required technical skills
- questions about relevant projects
- at least one problem-solving question

## Human Review
State clearly that the AI recommendation is advisory and the final hiring decision must be made by a human recruiter.
"""

    llm = create_recruitment_agent()

    response = llm.invoke(prompt)

    return response.content