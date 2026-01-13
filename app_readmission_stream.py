import streamlit as st
from dotenv import load_dotenv
from implementation.answer_readmission import answer_question

load_dotenv(override=True)

# Page config
st.set_page_config(
    page_title="CMS Readmission Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 CMS Readmission Assistant")
st.caption("Answers grounded exclusively in official CMS methodology documentation.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = []

# Create two columns
col1, col2 = st.columns([2, 1])

# LEFT COLUMN: Chat Interface
with col1:
    st.subheader("💬 Conversation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about CMS hospital-wide readmission measures..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer from RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching CMS documentation..."):
                answer, context = answer_question(prompt, st.session_state.messages[:-1])
                st.markdown(answer)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Update context
        st.session_state.context = context
        
        # Force rerun to update right column
        st.rerun()

# RIGHT COLUMN: Retrieved Context
with col2:
    st.subheader("📄 Retrieved Context")
    
    if st.session_state.context:
        for i, doc in enumerate(st.session_state.context):
            source = doc.metadata.get('file_name', 'Unknown Source')
            page = doc.metadata.get('page', 'N/A')
            
            with st.expander(f"📑 Chunk {i+1} (Page: {page})", expanded=(i == 0)):
                st.caption(f"**Source:** {source}")
                st.markdown(doc.page_content)
                st.divider()
    else:
        st.info("*Retrieved chunks from the CMS PDF will appear here after you ask a question.*")

# Add a clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.context = []
        st.rerun()
