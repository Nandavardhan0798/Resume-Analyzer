from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import streamlit as st
import tempfile

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

st.title = ("Resume Analyser")

# ----------------------------
# Load PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    doc = PDFPlumberLoader(tmp_file_path)
    pages = doc.load()
    texts = [d.page_content for d in pages]
    full_text = "\n".join(texts)

    # ----------------------------
    # Step 1: Check if resume
    # ----------------------------
    keywords = ["resume", "cv", "skills", "education", "experience"]
    if not any(k in full_text.lower() for k in keywords):
        st.warning("⚠️ This document is NOT a resume.")
        exit()
    else:
        st.success("✅ Resume detected. You can now ask questions about it.\n")

    # ----------------------------
    # Step 2: Split text into chunks
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    chunks = splitter.create_documents(texts)

    # ----------------------------
    # Step 3: Embeddings + Vector Store
    # ----------------------------
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # ----------------------------
    # Step 4: Setup LLM
    # ----------------------------
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )

    # ----------------------------
    # Tool 1: Resume QA
    # ----------------------------
    resume_prompt = PromptTemplate(
        template="""
    System: You are a Resume Assistant. Always use the provided resume content. 
    If the answer is not in the resume, say "Not found in the resume".

    Resume Content:
    {content}

    Question: {question}

    Answer:""",
        input_variables=["content", "question"]
    )

    def resume(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        content = "\n\n".join(doc.page_content for doc in docs)
        return llm.predict(resume_prompt.format(content=content, question=question))

    # ----------------------------
    # Tool 2: ATS Scorer
    # ----------------------------
    def ats(job_desc: str) -> str:
        prompt = f"""
    System: You are an ATS evaluator. Compare this resume with the job description.
    just Provide a match percentage (0-100%).
    Resume:
    {full_text}

    Job Description:
    {job_desc}

    Answer:"""
        return llm.predict(prompt)

    # ----------------------------
    # Tool 3: Resume Performance
    # ----------------------------
    def performance(_: str) -> str:
        prompt = f"""
    System: You are a professional career coach.
    Evaluate the resume for in just maximum 5 lines:
    - Formatting
    - Keyword usage
    - Clarity
    - Overall strength

    Provide a performance score (0-100%) and suggestions for improvement.

    Resume:
    {full_text}

    Answer:"""
        return llm.predict(prompt)

    # ----------------------------
    # Step 5: Register Tools
    # ----------------------------
    tools = [
        Tool(name="Resume_QA", func=resume, description="Answer questions about resume content."),
        Tool(name="ATS_Scorer", func=ats, description="Check resume against a job description."),
        Tool(name="Resume_Performance", func=performance, description="Evaluate resume formatting and effectiveness.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # ----------------------------
    # Step 6: Interactive loop
    # ----------------------------
    st.subheader("Paste your job discription and then ask your question about the resume (or type 'exit' to quit):")
    user_input = st.text_input("Enter your question here...").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Thank you, BYE...")
    if user_input == "":
        response = agent.run(user_input)
        st.write(f"AI: {response}\n")

    st.subheader("ATS tracer")
    job_desc = st.text_input("Enter your job discription here...").strip()
    if job_desc:
        ats_result = ats(job_desc)
        st.markdown(f"**ATS Analysis:**\n{ats_result}")

    st.subheader("📈 Resume Performance")
    if st.button("Evaluate Resume Performance"):
        performance_result = performance("Analyze")
        st.markdown(f"**Performance:**\n{performance_result}")
