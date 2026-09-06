import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from hr_assistant.document_loader import load_resume, documents_to_text
from hr_assistant.agent import analyze_candidate


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .footer-note {
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI HR Recruitment Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered resume screening, candidate matching and interview question generation"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------------

with st.expander("📌 About this Project", expanded=True):

    st.write(
        """
        This AI HR Recruitment Assistant helps recruiters evaluate
        candidates by comparing resumes with job descriptions.

        **Technologies used:**

        🤖 Agent  
        🛠️ Tools  
        🔎 RAG  
        🧠 Ollama + Qwen2.5  
        🗄️ Chroma Vector Database  
        🖥️ Streamlit
        """
    )


# ---------------------------------------------------------
# CANDIDATE EVALUATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📋 Candidate Evaluation</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# JOB DESCRIPTION
# ---------------------------------------------------------

with col1:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Enter the job description",
        height=350,
        placeholder=(
            "Paste the job description here..."
        ),
        label_visibility="collapsed",
    )


# ---------------------------------------------------------
# RESUME UPLOAD
# ---------------------------------------------------------

with col2:

    st.subheader("📄 Candidate Resume")

    resume_file = st.file_uploader(
        "Upload Candidate Resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX and TXT",
    )

    if resume_file is not None:

        st.success(
            f"✅ Resume uploaded: {resume_file.name}"
        )


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

st.markdown("---")

analyze_button = st.button(
    "🔍 Analyze Candidate",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if analyze_button:

    # Check Job Description

    if not job_description.strip():

        st.error(
            "⚠️ Please enter a Job Description."
        )

        st.stop()


    # Check Resume

    if resume_file is None:

        st.error(
            "⚠️ Please upload a candidate resume."
        )

        st.stop()


    # Create temporary resume file

    suffix = os.path.splitext(
        resume_file.name
    )[1]


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(
            resume_file.getbuffer()
        )

        temp_path = tmp.name


    try:

        # -------------------------------------------------
        # RUN ANALYSIS
        # -------------------------------------------------

        with st.spinner(
            "🤖 Reading resume and running AI recruitment analysis..."
        ):

            documents = load_resume(
                temp_path
            )

            resume_text = documents_to_text(
                documents
            )

            result = analyze_candidate(
                job_description=job_description,
                resume_text=resume_text,
            )


        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            "✅ Candidate analysis completed!"
        )


        # -------------------------------------------------
        # RECRUITMENT REPORT
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "📊 Recruitment Report"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(result)


        # -------------------------------------------------
        # EXTRACTED RESUME
        # -------------------------------------------------

        with st.expander(
            "📄 View Extracted Resume Text"
        ):

            st.text(resume_text)


        # -------------------------------------------------
        # AI COMPONENTS
        # -------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            "⚙️ AI Components Used"
            "</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)


        with c1:
            st.info("🤖 Agent")


        with c2:
            st.info("🛠️ Tools")


        with c3:
            st.info("🔎 RAG")


        with c4:
            st.info("🧠 Qwen2.5")


    except Exception as exc:

        st.error(
            f"❌ Something went wrong:\n\n{exc}"
        )


    finally:

        # Delete temporary file

        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
    ⚠️ <b>Human Review Required:</b>
    This AI assistant supports recruiters with candidate analysis.
    Final hiring decisions should always be made by a human recruiter.
    </div>
    """,
    unsafe_allow_html=True,
)