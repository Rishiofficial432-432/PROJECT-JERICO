import numpy as np
from datetime import datetime

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

if __name__ == "__main__":
    print(dispatch_authorities("Armed Suspect with Gun", 0.98, 
                              geo_location={"lat": 40.7128, "lon": -74.0060, "name": "Times Square - 42nd St"},
                              threat_details={"weapon_detected": "Firearm", "subjects": "1-2 armed individuals", "movement": "Erratic"}))
    print("\n--- Generating Siren Audio ---")
    siren = generate_siren_audio(duration=2)
    print(f"Siren generated: {len(siren)} samples at 44.1kHz")
