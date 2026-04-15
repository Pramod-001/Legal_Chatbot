import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MY_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.getenv(
    "DB_FAISS_PATH",
    os.path.join(BASE_DIR, "vectorstore", "db_faiss"),
)

if not MY_HF_TOKEN:
    raise RuntimeError(
        "Missing HUGGINGFACEHUB_API_TOKEN environment variable. "
        "Set it in your Render service environment settings."
    )

if not os.path.exists(DB_FAISS_PATH):
    raise RuntimeError(
        f"FAISS vectorstore not found at '{DB_FAISS_PATH}'. "
        "Ensure api_server/vectorstore/db_faiss is present in deployment."
    )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3})

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=MY_HF_TOKEN
)
chat_model = ChatHuggingFace(llm=llm)

system_prompt = (
    "You are Lexi Voice, a professional Indian Law AI assistant. "
    "Use the following legal context to answer the user's question. "
    "If user describes an incident/scenario, identify likely legal issues and provide practical next legal steps in India. "
    "Keep answers clear and actionable with this structure when relevant: "
    "1) Legal view, 2) Immediate steps, 3) Documents/Evidence to keep, 4) Where to file/approach, 5) Caution. "
    "Answer ONLY for the current user query. Never include any extra examples, extra contexts, or training artifacts. "
    "Never assume or invent facts (person, place, duration, prior attempts, evidence, relationship) not stated by user. "
    "If key facts are missing, say what can be done generally and ask concise clarifying questions. "
    "Never output tokens such as [/INST], [INST], Context:, Question:, or Answer:. "
    "Only mention exact section/article numbers when highly confident from reliable context; otherwise describe the legal principle without fabricating section numbers. "
    "Do not include any prefixes like 'user:' or 'lexi:'. "
    "Provide ONLY the direct answer. DO NOT repeat the question or include the word 'Answer:'. "
    "If unsure, say you don't know. \n\n Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

class Query(BaseModel):
    text: str


GREETING_PATTERNS = [
    r"^\s*hi+\s*[!.]?\s*$",
    r"^\s*hello+\s*[!.]?\s*$",
    r"^\s*hey+\s*[!.]?\s*$",
    r"^\s*good\s*(morning|afternoon|evening)\s*[!.]?\s*$",
]

LEGAL_KEYWORDS = {
    "law", "legal", "case", "court", "judge", "judgment", "petition", "fir",
    "bail", "arrest", "ipc", "crpc", "constitution", "rights", "contract",
    "agreement", "property", "tenant", "landlord", "divorce", "marriage",
    "maintenance", "alimony", "custody", "police", "cheque", "cheating",
    "fraud", "cybercrime", "consumer", "notice", "advocate", "lawyer",
    "writ", "appeal", "tribunal", "section", "act", "article", "amendment",
    "bill", "ordinance", "fundamental", "duty", "directive", "evidence",
    "complaint", "supreme", "high", "constitutional",
}

INCIDENT_HINT_WORDS = {
    "happened", "incident", "issue", "problem", "threat", "threatened", "abuse",
    "harass", "harassment", "blackmail", "scam", "stolen", "theft", "assault",
    "attack", "fight", "dispute", "refuse", "denied", "money", "loan", "salary",
    "employer", "company", "neighbor", "husband", "wife", "family", "property",
    "land", "rent", "tenant", "landlord", "police", "station", "notice",
}

NON_LEGAL_TOPICS = {
    "recipe", "cook", "cooking", "movie", "song", "music", "game", "cricket",
    "football", "weather", "temperature", "joke", "poem", "travel", "coding",
    "python", "javascript", "java", "bug", "gym", "workout", "diet",
}


def is_greeting(text: str) -> bool:
    normalized = text.strip().lower()
    return any(re.match(pattern, normalized) for pattern in GREETING_PATTERNS)


def is_legal_query(text: str) -> bool:
    normalized = text.lower().strip()
    words = re.findall(r"[a-zA-Z]+", normalized)

    if any(word in LEGAL_KEYWORDS for word in words):
        return True

    # Catch common legal shorthand with numbers, e.g. "article 370", "section 420", "ipc 302".
    legal_number_pattern = r"\b(article|section|ipc|crpc|constitution|act|amendment)\s*\d+[a-z]?\b"
    if re.search(legal_number_pattern, normalized):
        return True

    # Scenario-style legal descriptions may not include explicit legal keywords.
    if any(word in INCIDENT_HINT_WORDS for word in words):
        return True

    return False


def is_clearly_non_legal(text: str) -> bool:
    normalized = text.lower().strip()
    words = re.findall(r"[a-zA-Z]+", normalized)
    has_non_legal_topic = any(word in NON_LEGAL_TOPICS for word in words)
    has_legal_signal = is_legal_query(text)
    return has_non_legal_topic and not has_legal_signal


def clean_model_answer(answer: str) -> str:
    """Remove prompt artifacts and multi-context spillover from model output."""
    if not answer:
        return "I could not generate a proper response. Please try again."

    cleaned = answer.replace("[/INST]", " ").replace("[INST]", " ").strip()
    cleaned = re.sub(r"^\s*lexi\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*assistant\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*user\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    # Remove leading labels if they appear.
    cleaned = re.sub(r"^\s*(answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    # If model starts appending unrelated examples, keep only the first block.
    split_markers = [
        r"\n\s*Context\s*:",
        r"\n\s*Question\s*:",
        r"\n\s*Example\s*:",
        r"\n\s*User\s*:",
    ]
    for marker in split_markers:
        parts = re.split(marker, cleaned, maxsplit=1, flags=re.IGNORECASE)
        cleaned = parts[0].strip()

    # Remove any leftover artifact lines.
    cleaned_lines = []
    for line in cleaned.splitlines():
        low = line.strip().lower()
        if low.startswith("context:") or low.startswith("[/inst]") or low.startswith("[inst]"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()

    return cleaned or "I could not generate a proper legal response. Please rephrase your legal query."


def strip_prompt_echo(answer: str, user_text: str) -> str:
    """Remove echoed user prompt/question from model output."""
    if not answer:
        return answer

    cleaned = answer.strip()
    user_clean = re.sub(r"\s+", " ", user_text.strip().lower())
    ans_clean = re.sub(r"\s+", " ", cleaned.lower())

    # If model starts with the same prompt text, cut it off.
    if user_clean and ans_clean.startswith(user_clean):
        cleaned = cleaned[len(user_text.strip()):].lstrip(" :\n-")

    # Remove common Q/A wrappers if present.
    cleaned = re.sub(r"^\s*(question|query)\s*:\s*.*?\n+", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*(answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*assistant\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def build_guarded_query(user_text: str) -> str:
    """Inject strict constraints so model sticks only to provided facts."""
    return (
        "Use ONLY the facts provided by the user. "
        "Do not add new people/relationships, timeline, events, or evidence not explicitly mentioned. "
        "If a fact is missing, write 'Not provided by user' or ask a clarification question.\n\n"
        f"User facts: {user_text}"
    )


def remove_unprovided_assumptions(answer: str, user_text: str) -> str:
    """Remove common hallucinated relationship labels not present in user text."""
    normalized_user = user_text.lower()
    cleaned = answer

    replacements = {
        "neighbor": "other person",
        "neighbour": "other person",
        "landlord": "other person",
        "employer": "other person",
        "boss": "other person",
        "friend": "other person",
        "husband": "other person",
        "wife": "other person",
    }

    for term, repl in replacements.items():
        if term not in normalized_user:
            cleaned = re.sub(rf"\b{term}\b", repl, cleaned, flags=re.IGNORECASE)

    # If model prepends re-written user story lines, trim until legal structure starts.
    heading_match = re.search(r"(?:^|\n)\s*(1\)|Legal view:)", cleaned, flags=re.IGNORECASE)
    if heading_match:
        cleaned = cleaned[heading_match.start():].strip()

    return cleaned


def extract_legal_reference(user_text: str) -> str:
    """Extract focused legal reference like 'article 370' or 'section 144'."""
    match = re.search(
        r"\b(article|section|ipc|crpc|constitution)\s*(\d+[a-z]?)\b",
        user_text.lower()
    )
    if not match:
        return ""
    return f"{match.group(1)} {match.group(2)}"


def is_off_topic_for_reference(user_text: str, answer_text: str) -> bool:
    """If user asked about a specific legal reference, answer must include it."""
    target_ref = extract_legal_reference(user_text)
    if not target_ref:
        return False
    return target_ref not in answer_text.lower()


def regenerate_focused_answer(user_text: str) -> str:
    """Fallback generation without retrieval when RAG drifts off-topic."""
    focused_prompt = (
        "You are an Indian legal assistant. Answer ONLY this user query and stay strictly on topic. "
        "Do not add unrelated scenarios, people, or examples. "
        "If the query is a legal reference (like Article/Section), explain that exact provision in simple terms, "
        "its legal effect, and one short caution. "
        "Never output prefixes like 'Lexi:' or tokens like [INST]/[/INST].\n\n"
        f"User query: {user_text}"
    )
    llm_response = chat_model.invoke(focused_prompt)
    raw = getattr(llm_response, "content", str(llm_response))
    cleaned = clean_model_answer(raw)
    cleaned = remove_unprovided_assumptions(cleaned, user_text)
    return cleaned

@app.post("/chat")
async def chat_endpoint(query: Query):
    """Handles the AI logic when the 'Send' button is clicked."""
    try:
        user_text = query.text.strip()
        if not user_text:
            return {"answer": "Please share your legal question, and I will assist you."}

        if is_greeting(user_text):
            return {"answer": "Hi, this is Lexi Voice. How can I assist you with your legal issue today?"}

        if is_clearly_non_legal(user_text):
            return {"answer": "I am trained on legal data and can help only with legal questions. Please ask a legal query."}

        guarded_input = build_guarded_query(user_text)
        response = rag_chain.invoke({"input": guarded_input})
        answer_text = clean_model_answer(response["answer"])
        answer_text = strip_prompt_echo(answer_text, user_text)
        answer_text = remove_unprovided_assumptions(answer_text, user_text)

        # If RAG drifts for focused legal references (e.g., "Article 370"), regenerate a strict answer.
        if is_off_topic_for_reference(user_text, answer_text):
            answer_text = regenerate_focused_answer(user_text)
            answer_text = strip_prompt_echo(answer_text, user_text)
            
        return {"answer": answer_text}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
# .\venv\Scripts\Activate.ps1   (to activate virtual environment)
# uvicorn main:app --reload (to run backend)