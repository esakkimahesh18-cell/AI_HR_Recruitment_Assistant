from langchain_core.tools import tool


# Skills that our HR assistant can recognize
COMMON_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Pandas",
    "NumPy",
    "Git",
    "LangChain",
    "RAG",
    "FastAPI",
    "Docker",
    "AWS",
    "scikit-learn",
    "REST API",
    "Flask",
]


@tool
def extract_skills(text: str) -> list:
    """
    Extract skills that are explicitly present in the given text.
    Matching is case-insensitive.
    """

    text_lower = text.lower()

    found_skills = []

    for skill in COMMON_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(found_skills))


@tool
def calculate_match_score(
    required_skills: list,
    candidate_skills: list
) -> str:
    """
    Calculate deterministic skill match score.
    Only skills explicitly found in the resume are counted.
    """

    required = list(dict.fromkeys(required_skills))
    candidate = list(dict.fromkeys(candidate_skills))

    if not required:
        return "Match Score: 0% (No required skills found)"

    matched = [
        skill for skill in required
        if skill.lower() in [c.lower() for c in candidate]
    ]

    missing = [
        skill for skill in required
        if skill.lower() not in [c.lower() for c in candidate]
    ]

    score = (len(matched) / len(required)) * 100

    result = f"""
Match Score: {score:.1f}%

Matched Skills:
{chr(10).join("- " + skill for skill in matched) if matched else "- None"}

Missing Skills:
{chr(10).join("- " + skill for skill in missing) if missing else "- None"}
"""

    return result


@tool
def search_hr_knowledge(query: str) -> str:
    """
    Search the HR knowledge base using RAG.
    """

    from .rag import search_knowledge

    results = search_knowledge(query, k=4)

    if not results:
        return "No relevant HR knowledge found."

    output = []

    for result in results:
        source = result.metadata.get("source", "Unknown")
        content = result.page_content

        output.append(
            f"Source: {source}\n{content}"
        )

    return "\n\n".join(output)