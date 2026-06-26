import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


st.set_page_config(
    page_title="AI Career Copilot",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background-color: #0B1020;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 25px;
    color: white;
}

.user-message {
    background-color: #2563EB;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: white;
    font-size: 16px;
}

.bot-message {
    background-color: #1E293B;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: white;
    font-size: 16px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
}

</style>
""", unsafe_allow_html=True)


# LOAD EMBEDDINGS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LOAD VECTOR DB

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown(
    '<div class="main-title">🤖 AI Career Copilot</div>',
    unsafe_allow_html=True
)


for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
            🧑 {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-message">
            🤖 {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# CHAT INPUT

query = st.chat_input(
    "Ask anything about your resume, projects, SQL, DSA, AWS..."
)

#Queery

if query:

    # Save User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Retrieve Relevant Chunks
    docs = db.similarity_search(query, k=3)

    # Combine Context
    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # SECURE RAG PROMPT
    prompt = f"""
    You are an intelligent AI Career Assistant.

    STRICT RULES:
    1. Answer ONLY using the provided context.
    2. NEVER invent fake skills or experience.
    3. If information is missing, say:
       "I could not find this information in the documents."
    4. Be conversational, professional, and helpful.
    5. Give concise but informative answers.

    Context:
    {context}

    User Question:
    {query}

    Answer:
    """

    
    response = llm.invoke(prompt).content

    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    
    st.rerun()