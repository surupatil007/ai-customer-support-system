from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import json
import sqlite3
import os

from email_service import fetch_email_list, fetch_email_body, send_reply, save_config, load_config, test_connection

# ── Groq client ────────────────────────────────────────────────────────────
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_groq(prompt):
    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def query_model(prompt):
    try:
        return query_groq(prompt)
    except Exception as e:
        print("GROQ ERROR:", e)
        return "AI service error. Please try again."

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = pipeline("sentiment-analysis")

# ── Database ───────────────────────────────────────────────────────────────
conn = sqlite3.connect("data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    confidence REAL,
    embedding TEXT
)
""")
conn.commit()

cursor.execute("PRAGMA table_info(records)")
columns = [col[1] for col in cursor.fetchall()]
if "session_id" not in columns:
    cursor.execute("ALTER TABLE records ADD COLUMN session_id TEXT")
    conn.commit()

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "ICAP Backend Running"}


@app.post("/analyze")
def analyze(data: dict):
    text = data.get("text")
    session_id = data.get("session_id", "default")

    if not text:
        return {"error": "No text provided"}

    result = sentiment_analyzer(text)[0]
    sentiment = result["label"]
    confidence = round(result["score"], 3)

    vector_str = json.dumps(embedding_model.encode(text).tolist())
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, embedding) VALUES (?, ?, ?, ?)",
        (text, sentiment, confidence, vector_str)
    )
    conn.commit()

    return {"input_text": text, "sentiment": sentiment, "confidence": confidence}


@app.post("/search")
def search(data: dict):
    query = data.get("text")

    if not query:
        return {"error": "No query provided"}

    query_vector = embedding_model.encode(query)
    cursor.execute("SELECT id, text, embedding FROM records")
    rows = cursor.fetchall()

    results = []
    for row in rows:
        stored_vector = np.array(json.loads(row[2]))
        score = np.dot(query_vector, stored_vector)
        results.append({"text": row[1], "score": float(score)})

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results[:3]}


@app.post("/rag")
def rag(data: dict):
    query = data.get("text")

    if not query:
        return {"error": "No query provided"}

    query_vector = embedding_model.encode(query)
    session_id = data.get("session_id", "default")

    cursor.execute("SELECT text, embedding FROM records WHERE session_id=?", (session_id,))
    rows = cursor.fetchall()

    results = []
    for row in rows:
        stored_vector = np.array(json.loads(row[1]))
        score = np.dot(query_vector, stored_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
        )
        if score > 0.5:
            results.append((row[0], score))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    context = "\n".join([r[0] for r in results[:5]])

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

    result = sentiment_analyzer(query)[0]
    sentiment = result["label"]
    confidence = round(result["score"], 3)
    vector_str = json.dumps(embedding_model.encode(query).tolist())

    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, embedding, session_id) VALUES (?, ?, ?, ?, ?)",
        (query, sentiment, confidence, vector_str, session_id)
    )
    conn.commit()

    return {"query": query, "response": response_text, "context_used": context}


@app.get("/records")
def get_records():
    cursor.execute("SELECT * FROM records")
    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "text": row[1],
            "sentiment": row[2],
            "confidence": row[3],
            "session_id": row[5] if len(row) > 5 else "default"
        })

    return {"data": data}


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
    required = ["email", "password", "imap_host", "imap_port", "smtp_host", "smtp_port"]
    for field in required:
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
        emails = fetch_email_list(config, limit=limit)
        return {"status": "ok", "emails": emails}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/email/body/{email_id}")
def get_email_body(email_id: str):
    config = load_config()
    if not config:
        return {"status": "error", "message": "Email not configured."}
    try:
        body = fetch_email_body(config, email_id)
        return {"status": "ok", "body": body}
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

    query_vector = embedding_model.encode(body[:512])
    cursor.execute("SELECT text, embedding FROM records WHERE session_id=?", (session_id,))
    rows = cursor.fetchall()
    context_parts = []
    for row in rows:
        sv = np.array(json.loads(row[1]))
        score = float(np.dot(query_vector, sv) / (
            (np.linalg.norm(query_vector) * np.linalg.norm(sv)) + 1e-9))
        if score > 0.4:
            context_parts.append((row[0], score))
    context_parts.sort(key=lambda x: x[1], reverse=True)
    context = "\n".join(r[0] for r in context_parts[:5])

    prompt = EMAIL_PROMPT_TEMPLATE.format(body=body)
    if context:
        prompt = f"Relevant conversation history:\n{context}\n\n" + prompt

    ai_response = query_model(prompt)

    result = sentiment_analyzer(body[:512])[0]
    vector_str = json.dumps(embedding_model.encode(body[:512]).tolist())
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, embedding, session_id) VALUES (?,?,?,?,?)",
        (body, result["label"], round(result["score"], 3), vector_str, session_id)
    )
    conn.commit()

    send_status = "not_sent"
    if data.get("send", False) and from_addr:
        try:
            send_reply(config, from_addr, subject, ai_response, in_reply_to=message_id)
            send_status = "sent"
        except Exception as e:
            send_status = f"failed: {e}"

    return {
        "status": "ok",
        "response": ai_response,
        "context_used": context,
        "send_status": send_status,
    }


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
        send_reply(config, to_addr=to, subject=subj, body=body,
                   in_reply_to=data.get("message_id", ""))
        return {"status": "sent", "to": to}
    except Exception as e:
        import traceback
        print("SMTP ERROR:", traceback.format_exc())
        return {"status": "error", "message": str(e)}


# ── Auto-reply pipeline ────────────────────────────────────────────────────

_UNCERTAINTY_MARKERS = [
    "i don't have information",
    "i'm not sure",
    "i cannot help",
    "i can't help",
    "outside my knowledge",
    "i don't know",
    "unable to assist",
    "cannot determine",
    "no information available",
    "i lack the",
    "not within my",
    "please contact",
    "reach out to",
    "speak with a",
    "escalate",
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

    query_vector = embedding_model.encode(text[:512])
    cursor.execute("SELECT text, embedding FROM records WHERE session_id=?", (session_id,))
    rows = cursor.fetchall()
    context_parts = []
    for row in rows:
        sv = np.array(json.loads(row[1]))
        norm = np.linalg.norm(query_vector) * np.linalg.norm(sv) + 1e-9
        score = float(np.dot(query_vector, sv) / norm)
        if score > 0.4:
            context_parts.append((row[0], score))
    context_parts.sort(key=lambda x: x[1], reverse=True)
    context = "\n".join(r[0] for r in context_parts[:5])

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

    sentiment_result = sentiment_analyzer(text[:512])[0]
    vector_str = json.dumps(embedding_model.encode(text[:512]).tolist())
    cursor.execute(
        "INSERT INTO records (text, sentiment, confidence, embedding, session_id) VALUES (?,?,?,?,?)",
        (text, sentiment_result["label"], round(sentiment_result["score"], 3), vector_str, session_id)
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
