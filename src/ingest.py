import cv2
import sys
import time

# Default stream source - replace with your RTSP URL or camera index (e.g. 0)
STREAM_SOURCE = 0 

def get_stream(source=STREAM_SOURCE):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        sys.exit(1)
    
    print(f"Started ingesting from {source}...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame. Stream may have ended.")
                break
                
            # For demonstration, resizing frame to standard 640x480
            frame = cv2.resize(frame, (640, 480))
            
            # Send to next pipeline stage or display (commented out for headless)
            # cv2.imshow('Ingest', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            time.sleep(1/15.0)  # Default ~15 FPS fallback rate limiting
    except KeyboardInterrupt:
        print("Ingestion stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    get_stream()
