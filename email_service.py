import imaplib
import smtplib
import email as email_lib
import json
import os
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

CONFIG_FILE = "email_config.json"


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def _decode_str(value) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    result = ""
    for part, charset in parts:
        if isinstance(part, bytes):
            result += part.decode(charset or "utf-8", errors="ignore")
        else:
            result += str(part)
    return result


def _strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_body(msg) -> str:
    plain = ""
    html = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            if ct == "text/plain" and not plain:
                raw = part.get_payload(decode=True)
                if raw:
                    plain = raw.decode("utf-8", errors="ignore")
            elif ct == "text/html" and not html:
                raw = part.get_payload(decode=True)
                if raw:
                    html = raw.decode("utf-8", errors="ignore")
    else:
        ct = msg.get_content_type()
        raw = msg.get_payload(decode=True)
        if raw:
            decoded = raw.decode("utf-8", errors="ignore")
            if ct == "text/html":
                html = decoded
            else:
                plain = decoded

    if plain:
        return plain.strip()
    if html:
        return _strip_html(html)
    return ""


def _open_imap(config: dict):
    mail = imaplib.IMAP4_SSL(config["imap_host"], int(config["imap_port"]))
    mail.login(config["email"], config["password"])
    mail.select("INBOX")
    return mail


def test_connection(config: dict) -> dict:
    mail = _open_imap(config)
    _, ids = mail.search(None, "UNSEEN")
    count = len(ids[0].split()) if ids[0] else 0
    mail.logout()
    return {"unread": count}


def fetch_email_list(config: dict, limit: int = 20) -> list:
    """
    Fast header-only fetch — one IMAP round trip for all emails.
    Returns lightweight dicts suitable for the inbox list.
    Body is NOT included here; call fetch_email_body() when needed.
    """
    mail = _open_imap(config)

    _, ids = mail.search(None, "ALL")
    email_ids = ids[0].split() if ids[0] else []
    if not email_ids:
        mail.logout()
        return []

    recent = email_ids[-limit:]

    # Batch fetch all headers in a single IMAP command
    id_set = b",".join(recent)
    _, data = mail.fetch(id_set, "(RFC822.HEADER UID FLAGS)")
    mail.logout()

    emails = []
    i = 0
    while i < len(data):
        item = data[i]
        if not isinstance(item, tuple) or len(item) < 2:
            i += 1
            continue

        meta   = item[0].decode(errors="ignore") if isinstance(item[0], bytes) else str(item[0])
        raw_hdr = item[1]
        i += 1

        # Extract sequence number from meta string like "12 (RFC822.HEADER ...)"
        seq = meta.split()[0]

        msg = email_lib.message_from_bytes(raw_hdr)
        subject  = _decode_str(msg.get("Subject", "(No Subject)"))
        from_raw = msg.get("From", "")
        date_raw = msg.get("Date", "")
        msg_id   = msg.get("Message-ID", "")

        # Short preview from Subject since we have no body yet
        emails.append({
            "id":         seq,
            "from_addr":  from_raw,
            "to_addr":    msg.get("To", ""),
            "subject":    subject,
            "date":       date_raw,
            "body":       None,          # loaded on demand
            "message_id": msg_id,
        })

    # Return newest first
    emails.reverse()
    return emails


def fetch_email_body(config: dict, email_id: str) -> str:
    """Fetch full body of a single email by its sequence number."""
    mail = _open_imap(config)
    _, data = mail.fetch(email_id.encode(), "(RFC822)")
    mail.logout()

    if not data or not data[0] or not isinstance(data[0], tuple):
        return ""

    msg = email_lib.message_from_bytes(data[0][1])
    return _get_body(msg)[:5000]


def send_reply(config: dict, to_addr: str, subject: str, body: str,
               in_reply_to: str = "") -> None:
    import email.utils

    # Extract bare address from "Name <addr>" format — sendmail needs just the address
    _, bare_to = email.utils.parseaddr(to_addr)
    if not bare_to:
        raise ValueError(f"Could not parse recipient address: {to_addr!r}")

    msg = MIMEMultipart("alternative")
    msg["From"]    = config["email"]
    msg["To"]      = to_addr   # keep display name in header
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"]  = in_reply_to

    msg.attach(MIMEText(body, "plain"))

    port = int(config["smtp_port"])

    if port == 465:
        # Direct SSL (no STARTTLS)
        with smtplib.SMTP_SSL(config["smtp_host"], port, timeout=30) as server:
            server.login(config["email"], config["password"])
            server.sendmail(config["email"], bare_to, msg.as_string())
    else:
        # STARTTLS — port 587 (and fallback for anything else)
        with smtplib.SMTP(config["smtp_host"], port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(config["email"], config["password"])
            server.sendmail(config["email"], bare_to, msg.as_string())
