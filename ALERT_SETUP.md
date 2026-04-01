# 📱 SMS & Email Alert Setup Guide

## Quick Start

Your dashboard now sends **real-time alerts** to your phone when threats are detected! 

### Step 1: Set Environment Variables

Before running the dashboard, set your email credentials as environment variables:

```bash
# For macOS/Linux:
export ALERT_EMAIL="your-gmail@gmail.com"
export ALERT_PASSWORD="your-app-password"

# Then run the dashboard:
cd '/Users/admin/Documents/vs code/PICA/PROJECT-JERICO-main'
/opt/miniconda3/bin/python -m streamlit run src/dashboard.py --server.port=8507
```

### Step 2: Generate Gmail App Password

1. Go to https://myaccount.google.com/
2. Select **Security** (left sidebar)
3. Enable **2-Step Verification** (if not already enabled)
4. Go to **App passwords**
5. Select **Mail** and **Windows Computer** (or Mac)
6. Google will generate a 16-character password
7. Copy this password and use it as `ALERT_PASSWORD`

Example:
```bash
export ALERT_EMAIL="myemail@gmail.com"
export ALERT_PASSWORD="abcd efgh ijkl mnop"
```

### Step 3: Configure in Dashboard

When you open the dashboard:
1. Go to the sidebar **Alert Settings**
2. Enter your phone number: `+91-932-897-6799` ✓ (already saved)
3. Enter your email address for threat photos
4. Ensure **Enable Phone Alerts** toggle is ON

### How It Works

When a threat is detected:

1. 🔊 **Siren plays** on your laptop (immediate local alert)
2. 📱 **SMS sent** to your phone with:
   - Threat type & confidence
   - GPS coordinates (camera location)
   - Google Maps link
   
3. 📧 **Email sent** with:
   - Full threat analysis
   - Scene photo (threat image)
   - Camera details
   - Emergency response status

### SMS Gateway (Free, No API Needed)

Your Indian phone number (+91) will receive SMS via:
- **Airtel, Jio, Vodafone, BSNL** email-to-SMS gateways
- No API keys or paid services required
- Works with standard SMTP email

### Scene Photos

- Automatically saved when threats detected
- Sent to your email inbox
- Stored locally in `/tmp/threat_scene_*.jpg`

### Geolocation

The dashboard captures:
- 📍 **Camera GPS location** (manually set in sidebar)
- 🌐 **Browser geolocation** (device location where dashboard is running)
- All included in alert messages

### Troubleshooting

**SMS not received?**
- Check `ALERT_EMAIL` and `ALERT_PASSWORD` environment variables are set
- Verify Gmail 2-Step Verification and App Password
- Check phone is in service area
- Wait 30 seconds (can take time to deliver)

**Email not received?**
- Check spam folder
- Verify email address in dashboard sidebar
- Check SMTP credentials

**Photos not sent?**
- Scene photos auto-save to `/tmp/`
- Email must be configured (not default "your-email@gmail.com")
- Check disk space in `/tmp/`

### Test The Setup

```bash
# Test SMS manually:
python -c "
from src.alert import send_sms_alert
result = send_sms_alert('+91-932-897-6799', 'TEST ALERT', 0.95, 
                        {'lat': 40.7128, 'lon': -74.0060, 'name': 'Test Location'})
print(result)
"
```

---

**Your phone number:** +91-932-897-6799  
**Alert enabled:** Toggle in sidebar  
**Scene photos:** Automatically captured  
**Geolocation:** GPS from camera + browser location  

🚨 Stay safe!
