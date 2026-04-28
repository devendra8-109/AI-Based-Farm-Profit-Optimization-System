from pyngrok import ngrok
import time

try:
    # Open a HTTP tunnel on the default port 8501
    public_url = ngrok.connect(8501)
    print(f" * Public URL: {public_url}")
    print(" * Keep this script running to keep the link active.")
    
    while True:
        time.sleep(1)
except Exception as e:
    print(f"Error: {e}")
    print("Tip: You might need to set your authtoken using: ngrok config add-authtoken <your_token>")
