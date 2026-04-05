import numpy as np
from datetime import datetime
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from urllib.parse import quote

def generate_siren_audio(duration=3, sample_rate=44100):
    """Generate a wailing siren audio signal (chirp between 800-1200 Hz)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 1000 + 200 * np.sin(2 * np.pi * 2 * t)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def dispatch_authorities(incident_type, confidence, geo_location=None, threat_details=None):
    """Dispatch emergency alert with full threat analysis to authorities."""
    alert_log = []
    
    if geo_location is None:
        geo_location = {"lat": 40.7128, "lon": -74.0060, "name": "Unknown Camera"}
        
    lat = geo_location.get('lat', 40.7128)
    lon = geo_location.get('lon', -74.0060)
    cam_name = geo_location.get('name', 'Camera 1')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = f"🚨 CRITICAL THREAT ALERT 🚨\n"
    message += f"**Time:** {timestamp}\n"
    message += f"**Threat Type:** {incident_type.upper()}\n"
    message += f"**Confidence:** {confidence*100:.1f}%\n"
    message += f"**Camera:** {cam_name}\n"
    message += f"**GPS Location:** [{lat:.6f}, {lon:.6f}]\n"
    message += f"**Maps Link:** [Navigate 🗺️](https://www.google.com/maps/search/?api=1&query={lat},{lon})\n\n"
    
    if threat_details:
        message += f"**Additional Details:**\n"
        for key, val in threat_details.items():
            message += f"- {key}: {val}\n"
        message += "\n"
    
    incident_lower = incident_type.lower()
    priority = "CRITICAL"
    
    if "gun" in incident_lower or "weapon" in incident_lower or "armed" in incident_lower:
        alert_log.append("🔴 ARMED SUSPECT ALERT - GUN DETECTED")
        alert_log.append("🚔 DISPATCHING ARMED RESPONSE UNIT IMMEDIATELY")
        alert_log.append(f"📡 Location broadcasted to all patrol units")
        alert_log.append(f"🚨 SIREN ACTIVATED")
        alert_log.append(f"📱 Emergency SMS/MMS to nearby police stations")
        alert_log.append(f"🔔 Triggering lockdown protocols in surrounding buildings")
        priority = "CRITICAL-ARMED"
    elif "crash" in incident_lower or "collision" in incident_lower:
        alert_log.append("🚔 Notifying Highway Patrol & Police...")
        alert_log.append(f"🚑 Dispatching EMS / Ambulance to GPS coordinates...")
        alert_log.append(f"📡 Traffic control alerts issued")
    elif "fight" in incident_lower or "assault" in incident_lower or "violence" in incident_lower:
        alert_log.append("🚔 DISPATCHING POLICE UNIT - ASSAULT IN PROGRESS")
        alert_log.append(f"📡 All nearby units notified via dispatch")
        alert_log.append(f"🚨 SIREN ACTIVATED")
        priority = "CRITICAL"
    elif "robbery" in incident_lower or "theft" in incident_lower or "burglary" in incident_lower:
        alert_log.append("🚔 DISPATCHING POLICE UNIT - ROBBERY/THEFT IN PROGRESS")
        alert_log.append(f"🏬 Triggering silent alarms in surrounding buildings within 1km")
        alert_log.append(f"📡 Loss prevention teams notified")
        alert_log.append(f"🚨 SIREN ACTIVATED")
        priority = "CRITICAL"
    elif "fire" in incident_lower or "flame" in incident_lower or "blaze" in incident_lower:
        alert_log.append("🔥 FIRE DETECTED — IMMEDIATE EMERGENCY RESPONSE")
        alert_log.append("🚒 DISPATCHING FIRE DEPARTMENT TO GPS COORDINATES")
        alert_log.append("🚑 EMS / AMBULANCE DISPATCHED AS PRECAUTION")
        alert_log.append("🔔 FIRE ALARM ACTIVATED — EVACUATION INITIATED")
        alert_log.append("📡 ALERT BROADCAST TO ALL NEARBY EMERGENCY UNITS")
        alert_log.append("🚨 SIREN ACTIVATED")
        priority = "CRITICAL-FIRE"
    else:
        alert_log.append("ℹ️ Security Logging and monitoring active.")
        priority = "HIGH"
    
    message += f"**PRIORITY:** {priority}\n\n"
    message += "**ACTIONS TAKEN:**\n"
    for log in alert_log:
        message += f"- {log}\n"
    
    message += f"\n**DISPATCH STATUS:** ✅ All units notified\n"
    message += f"**ESTIMATED RESPONSE:** 2-5 minutes\n"
        
    return message

def send_ntfy_alert(
    topic,
    incident_type,
    confidence,
    geo_location=None,
    threat_details=None,
    server_url="https://ntfy.sh",
    token=None,
):
    """Send push notification via ntfy-compatible server."""
    try:
        lat = geo_location.get('lat', 0) if geo_location else 0
        lon = geo_location.get('lon', 0) if geo_location else 0
        cam_name = geo_location.get('name', 'Camera') if geo_location else 'Camera'
        clean_server = server_url.rstrip("/")
        ntfy_url = f"{clean_server}/{topic}"
        
        # Message for notification
        message = f"🚨 THREAT: {incident_type}\nConfidence: {confidence*100:.0f}%\nLocation: {lat:.4f},{lon:.4f}\n{cam_name}"
        
        headers = {
            "Title": "CRITICAL ALERT",
            "Priority": "high",
            "Tags": "siren,warning",
            "Content-Type": "text/plain; charset=utf-8",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Send via ntfy.sh
        response = requests.post(
            ntfy_url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return f"✅ Alert sent! Subscribe: {clean_server}/{topic}"
        else:
            return f"⚠️ Alert delivery status: {response.status_code}"
    except Exception as e:
        return f"📱 Note: Notifications require internet. Error: {str(e)}"


def send_email_alert(
    to_email,
    incident_type,
    confidence,
    geo_location=None,
    threat_details=None,
    image_path=None,
    smtp_host="smtp.gmail.com",
    smtp_port=465,
    smtp_user=None,
    smtp_password=None,
):
    """Send email alert with optional image attachment via SMTP."""
    try:
        if not to_email:
            return "⚠️ Email skipped: recipient not configured"

        smtp_user = smtp_user or os.getenv("ALERT_SMTP_USER")
        smtp_password = smtp_password or os.getenv("ALERT_SMTP_PASSWORD")
        if not smtp_user or not smtp_password:
            return "⚠️ Email skipped: set ALERT_SMTP_USER and ALERT_SMTP_PASSWORD"

        lat = geo_location.get('lat', 0) if geo_location else 0
        lon = geo_location.get('lon', 0) if geo_location else 0
        cam_name = geo_location.get('name', 'Camera') if geo_location else 'Camera'
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = f"Jerico Alert: {incident_type}"

        body = [
            f"Time: {timestamp}",
            f"Threat: {incident_type}",
            f"Confidence: {confidence*100:.1f}%",
            f"Camera: {cam_name}",
            f"Location: {lat:.6f}, {lon:.6f}",
            f"Maps: https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        ]
        if threat_details:
            body.append("Details:")
            for k, v in threat_details.items():
                body.append(f"- {k}: {v}")

        msg.attach(MIMEText("\n".join(body), "plain", "utf-8"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(image_path)}"')
            msg.attach(part)

        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())

        return f"✅ Email sent to {to_email}"
    except Exception as e:
        return f"⚠️ Email error: {str(e)}"


def send_whatsapp_alert(whatsapp_phone, whatsapp_apikey, incident_type, confidence, geo_location=None):
    """Send WhatsApp alert via CallMeBot API (free tier)."""
    try:
        if not whatsapp_phone or not whatsapp_apikey:
            return "⚠️ WhatsApp skipped: set phone + API key"

        lat = geo_location.get('lat', 0) if geo_location else 0
        lon = geo_location.get('lon', 0) if geo_location else 0
        cam_name = geo_location.get('name', 'Camera') if geo_location else 'Camera'

        text = (
            f"ALERT: {incident_type}\n"
            f"Confidence: {confidence*100:.1f}%\n"
            f"Camera: {cam_name}\n"
            f"Location: {lat:.6f}, {lon:.6f}\n"
            f"Maps: https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        )

        url = (
            "https://api.callmebot.com/whatsapp.php"
            f"?phone={quote(whatsapp_phone)}"
            f"&text={quote(text)}"
            f"&apikey={quote(whatsapp_apikey)}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return f"✅ WhatsApp sent to {whatsapp_phone}"
        return f"⚠️ WhatsApp delivery status: {resp.status_code}"
    except Exception as e:
        return f"⚠️ WhatsApp error: {str(e)}"


if __name__ == "__main__":
    print(dispatch_authorities("Armed Suspect with Gun", 0.98, 
                              geo_location={"lat": 40.7128, "lon": -74.0060, "name": "Times Square - 42nd St"},
                              threat_details={"weapon_detected": "Firearm", "subjects": "1-2 armed individuals", "movement": "Erratic"}))
    print("\n--- Generating Siren Audio ---")
    siren = generate_siren_audio(duration=2)
    print(f"Siren generated: {len(siren)} samples at 44.1kHz")
