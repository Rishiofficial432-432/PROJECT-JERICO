def dispatch_authorities(incident_type, confidence, geo_location=None):
    alert_log = []
    
    if geo_location is None:
        geo_location = {"lat": 40.7128, "lon": -74.0060, "name": "Unknown Camera"}
        
    lat = geo_location.get('lat', 40.7128)
    lon = geo_location.get('lon', -74.0060)
    cam_name = geo_location.get('name', 'Camera 1')
    
    message = f"🚨 EMERGENCY DISPATCH 🚨\n"
    message += f"**Incident:** {incident_type.upper()} ({confidence*100:.1f}% confidence)\n"
    message += f"**Location:** {cam_name}\n"
    message += f"**GPS Pinpoint:** [Navigate via Google Maps ⮞](https://www.google.com/maps/search/?api=1&query={lat},{lon})\n\n"
    
    incident_lower = incident_type.lower()
    
    if "crash" in incident_lower or "collision" in incident_lower:
        alert_log.append("🚔 Notifying Highway Patrol & Police...")
        alert_log.append(f"🚑 Dispatching EMS / Ambulance to GPS coordinates...")
    elif "fight" in incident_lower or "assault" in incident_lower:
        alert_log.append("🚔 Dispatching Police Unit immediately...")
    elif "robbery" in incident_lower or "theft" in incident_lower or "burglary" in incident_lower:
        alert_log.append("🚔 Dispatching Police Unit immediately...")
        alert_log.append(f"🏬 Triggering alert to local shop owners within 500m of scene...")
    else:
        alert_log.append("ℹ️ Security Logging Only.")
        
    for log in alert_log:
        message += f"- {log}\n"
        
    return message

if __name__ == "__main__":
    print(dispatch_authorities("a car crash or traffic collision", 0.98))
