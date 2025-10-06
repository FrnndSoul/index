# resources/email_notifier.py
#!/usr/bin/env python3
import os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USER", "saulfernandemil@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASS", "tikl bpnp yayx xxxb")
SMTP_SENDER   = os.getenv("SMTP_SENDER", SMTP_USERNAME)
ALERT_EMAILS  = [e.strip() for e in os.getenv("ALERT_EMAILS", "saulfernandemil@gmail.com").split(",") if e.strip()]

def send_email(subject: str, html_body: str, image_bytes: bytes | None = None):
    msg = MIMEMultipart()
    msg["From"] = SMTP_SENDER
    msg["To"] = ", ".join(ALERT_EMAILS)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    if image_bytes:
        msg.attach(MIMEImage(image_bytes, name="snapshot.jpg"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        if SMTP_USERNAME and SMTP_PASSWORD:
            s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.sendmail(SMTP_SENDER, ALERT_EMAILS, msg.as_string())
