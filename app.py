
import streamlit as st

from config import (
    PDF_PATH,
    NO_INFO,
    LOGIN_FAILED,
    SESSION_TIMEOUT_SEC
)

from auth import (
    initialize_auth_state,
    authenticate_user,
    logout_user,
    get_current_user,
    is_authenticated,
    check_session_expiry
)

from permissions import get_allowed_levels

from audit import audit

from rag.ingestion import process_pdf
from rag.vector_store import VectorStore
from rag.retrieval import retrieve_context
from rag.llm import generate_answer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nexora | Secure Multimodal RAG",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. SESSION INITIALIZATION
# ============================================================

initialize_auth_state()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "session_expired" not in st.session_state:
    st.session_state.session_expired = False

check_session_expiry()


# ============================================================
# 3. LOAD RAG SYSTEM
# ============================================================

@st.cache_resource(show_spinner="Loading Secure Multimodal RAG...")

def load_rag_system():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # Process PDF
    data = process_pdf(PDF_PATH)

    # Build FAISS indexes
    vector_store = VectorStore()

    vector_store.build(
        data["text_docs"],
        data["image_docs"],
        data["image_vecs"]
    )

    return {
        "vector_store": vector_store,
        "image_store": data["image_store"]
    }


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #151922;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }

    .sub-title {
        color: #9ca3af;
        font-size: 14px;
    }

    .user-card {
        background-color: #202938;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }

    .login-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown("## 🔐 Nexora")

        st.caption("Secure Multimodal RAG")

        st.divider()

        user = get_current_user()

        if user:

            st.success("Authenticated")

            st.markdown(
                f"""
                **{user['name']}**

                Role: `{user['role'].upper()}`
                """
            )

            st.markdown("**Accessible classifications**")

            allowed = get_allowed_levels(user["role"])

            for level in sorted(allowed):
                st.write(f"✓ {level}")

            if st.button(
                "Logout",
                use_container_width=True
            ):

                audit(user, "logout")

                logout_user()

                st.rerun()

        else:

            st.info("Anonymous access")

            st.markdown("### Sign in")

            with st.form("login_form"):

                username = st.text_input(
                    "Username"
                )

                password = st.text_input(
                    "Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True
                )

                if submitted:

                    if not username or not password:

                        st.error(
                            "Enter both username and password."
                        )

                    else:

                        authenticated_user, error = authenticate_user(
                            username,
                            password
                        )

                        if authenticated_user:

                            audit(
                                authenticated_user,
                                "login_success"
                            )

                            st.success("Login successful!")

                            st.rerun()

                        else:

                            audit(
                                None,
                                "login_failed",
                                decision="authentication failed"
                            )

                            st.error(error)

        st.divider()

        if st.button(
            "New conversation",
            use_container_width=True
        ):

            st.session_state.chat_history = []
            st.session_state.pending_query = None

            st.rerun()

        st.caption("Nexora Secure Document Assistant")


# ============================================================
# 6. LOGIN PAGE
# ============================================================

def render_login_message():

    st.markdown(
        '<div class="login-title">🔐 Nexora</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#9ca3af'>"
        "Secure Multimodal Document Assistant"
        "</p>",
        unsafe_allow_html=True
    )

    st.info(
        "You can ask questions about public company information "
        "without signing in. Sign in when additional access is required."
    )


# ============================================================
# 7. CHAT HEADER
# ============================================================

def render_header():

    st.markdown(
        '<div class="main-title">Nexora AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Ask questions about company documents using text and image retrieval.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


# ============================================================
# 8. PROCESS USER QUERY
# ============================================================

def process_question(query, rag):

    user = get_current_user()

    role = user["role"] if user else "anonymous"

    allowed = get_allowed_levels(role)

    # Secure retrieval
    result = retrieve_context(
        query=query,
        vector_store=rag["vector_store"],
        allowed=allowed,
        image_store=rag["image_store"]
    )

    # Restricted content is a reason to request login
    if result["requires_login"] and user is None:

        st.session_state.pending_query = query

        audit(
            user,
            "auth_challenge",
            query,
            decision="restricted content matched"
        )

        return {
            "answer": LOGIN_FAILED,
            "requires_login": True
        }

    # No accessible content
    if not result["has_content"]:

        audit(
            user,
            "no_answer",
            query,
            decision="nothing accessible"
        )

        return {
            "answer": NO_INFO,
            "requires_login": False
        }

    # Generate answer using only authorized content
    answer = generate_answer(
        query=query,
        docs=result["documents"],
        image_ids=result["image_ids"],
        image_store=rag["image_store"]
    )

    # Record sources
    sources = [
        f"Page {doc.metadata['page'] + 1}"
        for doc in result["documents"]
    ]

    sources.extend(result["image_ids"])

    audit(
        user,
        "answered",
        query,
        sources=sources,
        decision="allowed"
    )

    return {
        "answer": answer,
        "requires_login": False,
        "documents": result["documents"],
        "image_ids": result["image_ids"]
    }


# ============================================================
# 9. DISPLAY ANSWER
# ============================================================

def display_answer(result):

    st.markdown(result["answer"])

    if result.get("documents"):

        with st.expander("📄 Sources"):

            for doc in result["documents"]:

                st.markdown(
                    f"**Page {doc.metadata['page'] + 1}**"
                )

                st.caption(
                    doc.page_content[:350]
                )

    if result.get("image_ids"):

        with st.expander("🖼️ Retrieved images"):

            for image_id in result["image_ids"]:

                image_bytes = rag["image_store"][image_id]

                import base64

                st.image(
                    base64.b64decode(image_bytes),
                    caption=image_id,
                    use_container_width=True
                )


# ============================================================
# 10. MAIN APPLICATION
# ============================================================

render_sidebar()

render_login_message()

render_header()

try:

    rag = load_rag_system()

except Exception as error:

    st.error(
        f"Could not initialize RAG system: {error}"
    )

    st.stop()


# Display previous messages
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Handle a question waiting for authentication
pending_query = st.session_state.pending_query

if pending_query and is_authenticated():

    st.session_state.pending_query = None



    with st.chat_message("assistant"):

        with st.spinner("Searching authorized documents..."):

            result = process_question(
                pending_query,
                rag
            )

            if result["requires_login"]:

                st.warning(
                    "Please sign in to access this information."
                )

            else:

                display_answer(result)

                st.session_state.chat_history.extend([
                    {
                        "role": "user",
                        "content": pending_query
                    },
                    {
                        "role": "assistant",
                        "content": result["answer"]
                    }
                ])


# Chat input
query = st.chat_input(
    "Ask something about your company documents..."
)

if query:

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = process_question(query, rag)

                if result["requires_login"]:

                    st.warning(
                        "🔒 Additional verification is required. "
                        "Please sign in using the sidebar. "
                        "Your question will be continued after login."
                    )

                else:

                    display_answer(result)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"]
                    })

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )

st.caption(
    "Nexora | Secure Multimodal RAG | "
    "Access-controlled document retrieval"
)
