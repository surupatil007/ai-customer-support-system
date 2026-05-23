from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
import os

from email_service import fetch_email_list, fetch_email_body, send_reply, save_config, load_config, test_connection

# ── Groq ───────────────────────────────────────────────────────────────────
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_model(prompt):
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("GROQ ERROR:", e)
        return "AI service error. Please try again."

# ── Sentiment (VADER — no model download) ─────────────────────────────────
_vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    compound = _vader.polarity_scores(text)["compound"]
    label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
    return label, round(abs(compound), 3)

# ── Similarity search (TF-IDF — no torch) ─────────────────────────────────
def find_similar(query, texts, top_n=5, threshold=0.1):
    if not texts:
        return []
    docs = [query] + list(texts)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
    except ValueError:
        return []
    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
    results = [(texts[i], float(scores[i])) for i in range(len(texts)) if scores[i] > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ───────────────────────────────────────────────────────────────
conn = sqlite3.connect("data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT,
    sentiment   TEXT,
    confidence  REAL,
    session_id  TEXT
)
""")
conn.commit()

# keep backward compat with older DBs that had session_id added as a migration
cursor.execute("PRAGMA table_info(records)")
if "session_id" not in [col[1] for col in cursor.fetchall()]:
    cursor.execute("ALTER TABLE records ADD COLUMN session_id TEXT")
    conn.commit()

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "ICAP Backend Running"}


@app.post("/analyze")
def analyze(data: dict):
    text       = data.get("text")
    session_id = data.get("session_id", "default")

    if not text:
        return {"error": "No text provided"}

    sentiment, confidence = analyze_sentiment(text)
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, session_id) VALUES (?,?,?,?)",
        (text, sentiment, confidence, session_id)
    )
    conn.commit()
    return {"input_text": text, "sentiment": sentiment, "confidence": confidence}


@app.post("/search")
def search(data: dict):
    query = data.get("text")
    if not query:
        return {"error": "No query provided"}

    cursor.execute("SELECT text FROM records")
    texts = [row[0] for row in cursor.fetchall()]
    results = find_similar(query, texts, top_n=3, threshold=0.0)
    return {"results": [{"text": t, "score": s} for t, s in results]}


@app.post("/rag")
def rag(data: dict):
    query      = data.get("text")
    session_id = data.get("session_id", "default")

    if not query:
        return {"error": "No query provided"}

    cursor.execute("SELECT text FROM records WHERE session_id=?", (session_id,))
    texts   = [row[0] for row in cursor.fetchall()]
    similar = find_similar(query, texts, threshold=0.1)
    context = "\n".join(t for t, _ in similar)

    prompt = f"""
You are a professional customer support assistant.

IMPORTANT:
- Answer based on conversation history
- Do NOT mix unrelated topics
- Keep response short and clear

Conversation History:
{context}

Current User Query:
{query}

Final Answer:
"""
    response_text = query_model(prompt)

    sentiment, confidence = analyze_sentiment(query)
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, session_id) VALUES (?,?,?,?)",
        (query, sentiment, confidence, session_id)
    )
    conn.commit()
    return {"query": query, "response": response_text, "context_used": context}


@app.get("/records")
def get_records():
    cursor.execute("SELECT id, text, sentiment, confidence, session_id FROM records")
    return {"data": [
        {"id": r[0], "text": r[1], "sentiment": r[2], "confidence": r[3], "session_id": r[4] or "default"}
        for r in cursor.fetchall()
    ]}


@app.get("/stats")
def stats():
    cursor.execute("SELECT COUNT(DISTINCT session_id) FROM records")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM records WHERE sentiment='POSITIVE'")
    positive = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM records WHERE sentiment='NEGATIVE'")
    negative = cursor.fetchone()[0]
    return {"total": total, "positive": positive, "negative": negative}


# ── Email integration ──────────────────────────────────────────────────────

EMAIL_PROMPT_TEMPLATE = """
You are a professional customer support assistant.
Read the following customer email and write a clear, helpful, and concise reply.
Do NOT include a subject line. Start directly with the greeting.

Customer Email:
{body}

Support Reply:
"""


@app.post("/email/connect")
def connect_email(data: dict):
    for field in ["email", "password", "imap_host", "imap_port", "smtp_host", "smtp_port"]:
        if not data.get(field):
            return {"status": "error", "message": f"Missing field: {field}"}
    try:
        info = test_connection(data)
        save_config(data)
        return {"status": "connected", "unread": info["unread"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/email/status")
def email_status():
    config = load_config()
    if not config:
        return {"connected": False}
    try:
        info = test_connection(config)
        return {"connected": True, "email": config.get("email"), "unread": info["unread"]}
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/email/fetch")
def get_emails(limit: int = 20):
    config = load_config()
    if not config:
        return {"status": "error", "message": "Email not configured. Use /email/connect first."}
    try:
        return {"status": "ok", "emails": fetch_email_list(config, limit=limit)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/email/body/{email_id}")
def get_email_body(email_id: str):
    config = load_config()
    if not config:
        return {"status": "error", "message": "Email not configured."}
    try:
        return {"status": "ok", "body": fetch_email_body(config, email_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/email/auto-reply")
def auto_reply_email(data: dict):
    config = load_config()
    if not config:
        return {"status": "error", "message": "Email not configured."}

    body       = data.get("body", "").strip()
    from_addr  = data.get("from_addr", "")
    subject    = data.get("subject", "")
    message_id = data.get("message_id", "")

    if not body:
        return {"status": "error", "message": "Email body is empty."}

    session_id = f"email-{from_addr}"
    cursor.execute("SELECT text FROM records WHERE session_id=?", (session_id,))
    texts   = [row[0] for row in cursor.fetchall()]
    similar = find_similar(body[:512], texts, threshold=0.1)
    context = "\n".join(t for t, _ in similar)

    prompt = EMAIL_PROMPT_TEMPLATE.format(body=body)
    if context:
        prompt = f"Relevant conversation history:\n{context}\n\n" + prompt

    ai_response = query_model(prompt)

    sentiment, confidence = analyze_sentiment(body[:512])
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, session_id) VALUES (?,?,?,?)",
        (body, sentiment, confidence, session_id)
    )
    conn.commit()

    send_status = "not_sent"
    if data.get("send", False) and from_addr:
        try:
            send_reply(config, from_addr, subject, ai_response, in_reply_to=message_id)
            send_status = "sent"
        except Exception as e:
            send_status = f"failed: {e}"

    return {"status": "ok", "response": ai_response, "context_used": context, "send_status": send_status}


@app.post("/email/send")
def send_manual_email(data: dict):
    config = load_config()
    if not config:
        return {"status": "error", "message": "Email not configured."}
    to   = data.get("to", "").strip()
    body = data.get("body", "").strip()
    subj = data.get("subject", "").strip()
    if not to:
        return {"status": "error", "message": "Recipient address is empty."}
    if not body:
        return {"status": "error", "message": "Reply body is empty."}
    try:
        send_reply(config, to_addr=to, subject=subj, body=body, in_reply_to=data.get("message_id", ""))
        return {"status": "sent", "to": to}
    except Exception as e:
        import traceback
        print("SMTP ERROR:", traceback.format_exc())
        return {"status": "error", "message": str(e)}


# ── Auto-reply pipeline ────────────────────────────────────────────────────

_UNCERTAINTY_MARKERS = [
    "i don't have information", "i'm not sure", "i cannot help", "i can't help",
    "outside my knowledge", "i don't know", "unable to assist", "cannot determine",
    "no information available", "i lack the", "not within my", "please contact",
    "reach out to", "speak with a", "escalate",
]

AUTO_REPLY_PROMPT = """\
You are a professional customer support AI assistant.

INSTRUCTIONS:
- Answer the customer's message clearly and helpfully.
- Use the context below if it is relevant.
- If you are confident you can resolve the issue, start your reply with: HANDLE:
- If the issue is out of scope, requires account access, is a legal/fraud matter,
  or you genuinely cannot resolve it, start your reply with: ESCALATE: <one-line reason>

Context from knowledge base:
{context}

Customer message:
{message}

Reply:"""


@app.post("/auto-reply")
def auto_reply_pipeline(data: dict):
    text       = (data.get("text") or "").strip()
    session_id = data.get("session_id", "default")
    from_addr  = data.get("from_addr", "")
    subject    = data.get("subject", "")
    message_id = data.get("message_id", "")
    do_send    = data.get("send", False)

    if not text:
        return {"status": "error", "message": "Empty message"}

    cursor.execute("SELECT text FROM records WHERE session_id=?", (session_id,))
    texts   = [row[0] for row in cursor.fetchall()]
    similar = find_similar(text[:512], texts, threshold=0.1)
    context = "\n".join(t for t, _ in similar)

    prompt = AUTO_REPLY_PROMPT.format(context=context or "No relevant context found.", message=text)
    raw    = query_model(prompt)

    can_handle   = True
    route_reason = ""
    response     = raw.strip()

    if response.upper().startswith("ESCALATE:"):
        can_handle   = False
        parts        = response.split("\n", 1)
        route_reason = parts[0][9:].strip()
        response     = parts[1].strip() if len(parts) > 1 else "This issue has been escalated to a human agent."
    elif response.upper().startswith("HANDLE:"):
        response = response[7:].strip()
    else:
        lower = response.lower()
        for marker in _UNCERTAINTY_MARKERS:
            if marker in lower:
                can_handle   = False
                route_reason = "AI response indicates uncertainty"
                break
        if not context.strip():
            can_handle   = False
            route_reason = "No relevant knowledge base context found for this query"

    confidence = 85 if can_handle else 20
    if context and not can_handle:
        confidence = 35

    sentiment, conf = analyze_sentiment(text[:512])
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, session_id) VALUES (?,?,?,?)",
        (text, sentiment, conf, session_id)
    )
    conn.commit()

    send_status = "not_sent"
    if do_send and can_handle and from_addr:
        config = load_config()
        if config:
            try:
                send_reply(config, from_addr, subject, response, in_reply_to=message_id)
                send_status = "sent"
            except Exception as e:
                send_status = f"failed: {e}"
        else:
            send_status = "no_email_config"

    return {
        "status":       "ok",
        "response":     response,
        "context_used": context,
        "can_handle":   can_handle,
        "confidence":   confidence,
        "route_reason": route_reason,
        "send_status":  send_status,
    }
