import streamlit as st
import tempfile
from typing_extensions import TypedDict

# LangChain / RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, START, END


# ================================
# PAGE UI
# ================================
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🤖 AI Resume Screening System")
st.write("Upload a resume + paste job description to screen candidate.")


# ================================
# FRONTEND API KEY INPUT
# ================================
groq_key = st.text_input(
    "🔑 Paste your Groq API Key",
    type="password",
    placeholder="gsk_xxxxxxxxx"
)

if not groq_key:
    st.warning("Please paste your Groq API key to continue.")
    st.stop()


# ================================
# LLM SETUP
# ================================
llm = ChatGroq(
    groq_api_key=groq_key.strip(),
    model_name="llama-3.1-8b-instant"
)


# ================================
# JOB DESCRIPTION INPUT
# ================================
job_description = st.text_area(
    "🧾 Paste Job Description",
    height=200,
    placeholder="Paste role requirements, skills, experience..."
)

if not job_description:
    st.warning("Please paste job description.")
    st.stop()


# ================================
# STATE SCHEMA
# ================================
class State(TypedDict):
    application: str
    job_description: str
    resume_context: str
    experience_level: str
    skill_match: str
    response: str


# ================================
# BUILD RAG PIPELINE
# ================================
def build_retriever(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    resume_text = " ".join([doc.page_content for doc in documents])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.create_documents([resume_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ================================
# LANGGRAPH NODES
# ================================
def retrieve_resume_context(state: State, retriever):
    # Use job description to search relevant resume sections
    query = state["job_description"]
    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])
    return {"resume_context": context}


def categorize_experience(state: State):

    prompt = ChatPromptTemplate.from_template(
        """
        Based on the resume context below, categorize the candidate as:
        'Entry-level', 'Mid-Level', or 'Senior-level'.

        Resume Context:
        {context}
        """
    )

    chain = prompt | llm
    result = chain.invoke({"context": state["resume_context"]}).content

    return {"experience_level": result}


def assess_skillset(state: State):

    prompt = ChatPromptTemplate.from_template(
        """
        You are an ATS screening system.

        Compare the resume against the job description.

        Job Description:
        {job}

        Resume Context:
        {context}

        Instructions:
        1. Extract REQUIRED technical skills from job description.
        2. Check if those skills exist in resume.
        3. If major required skills are missing → respond 'no match'.
        4. Only respond 'match' if majority of required tech stack exists.
        5. Be STRICT. Do not assume transferable skills.

        Respond ONLY one word:
        match
        OR
        no match
        """
    )

    chain = prompt | llm
    result = chain.invoke({
        "job": state["job_description"],
        "context": state["resume_context"]
    }).content.strip().lower()

    return {"skill_match": result}



def schedule_hr_interview(state: State):
    return {"response": "✅ Candidate shortlisted for HR interview."}


def escalate_to_recruiter(state: State):
    return {"response": "⚠️ Senior candidate but skill mismatch → escalate to recruiter."}


def reject_application(state: State):
    return {"response": "❌ Candidate rejected."}


# ================================
# ROUTING LOGIC
# ================================
def route_app(state: State):
    skill = state["skill_match"].strip().lower()
    exp = state["experience_level"].strip().lower()

    if skill == "match":
        return "schedule_hr_interview"
    elif exp == "senior-level" and skill == "no match":
        return "escalate_to_recruiter"
    else:
        return "reject_application"



# ================================
# BUILD GRAPH
# ================================
def build_workflow(retriever):

    workflow = StateGraph(State)

    def retrieve_node(state):
        return retrieve_resume_context(state, retriever)

    workflow.add_node("retrieve_resume_context", retrieve_node)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "retrieve_resume_context")
    workflow.add_edge("retrieve_resume_context", "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")

    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


# ================================
# FILE UPLOAD UI
# ================================
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("Resume uploaded successfully!")

    if st.button("🚀 Screen Candidate"):

        with st.spinner("Analyzing resume..."):

            retriever = build_retriever(pdf_path)
            app = build_workflow(retriever)

            result = app.invoke({
                "application": "screen candidate",
                "job_description": job_description
            })

            st.subheader("📊 Screening Result")
            st.success(result["response"])
