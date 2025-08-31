import cv2
import cvzone
import time
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# ----------------- CONFIG -----------------
confidence = 0.6
model = YOLO("models/m_version_1_30.pt")
classNames = ["fake", "real"]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_frame_time = 0
current_label = "Waiting..."
current_conf = 0
fps_value = 0

# ----------------- FLASK APP -----------------
app = Flask(__name__)

# HTML template (dark modern dashboard)
html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Anti-Spoofing Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background-color: #0d1117;
      color: #f0f6fc;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      text-align: center;
      padding: 20px;
    }
    h1 {
      margin-bottom: 20px;
      color: #58a6ff;
    }
    .video-container {
      display: flex;
      justify-content: center;
      margin-bottom: 20px;
    }
    .status-box {
      font-size: 2rem;
      font-weight: bold;
      padding: 15px;
      border-radius: 10px;
      display: inline-block;
    }
    .status-real {
      background-color: #198754;
      color: white;
    }
    .status-fake {
      background-color: #dc3545;
      color: white;
    }
    .status-waiting {
      background-color: #6c757d;
      color: white;
    }
    .info-box {
      margin-top: 15px;
      font-size: 1.1rem;
      color: #adb5bd;
    }
    footer {
      margin-top: 30px;
      font-size: 0.9rem;
      color: #6c757d;
    }
  </style>
</head>
<body>
  <h1>AI Anti-Spoofing System</h1>
  <div class="video-container">
    <img src="{{ url_for('video_feed') }}" width="640" height="480" class="rounded shadow">
  </div>
  <div>
    <span id="status" class="status-box status-waiting">Waiting...</span>
  </div>
  <div class="info-box">
    <p id="confidence">Confidence: 0%</p>
    <p id="fps">FPS: 0</p>
    <p>Model: YOLOv8 Custom (Anti-Spoofing)</p>
  </div>
  <footer>
    &copy; 2025 Anti-Spoofing Project | Powered by Flask + YOLO
  </footer>

  <script>
    async function updateStatus() {
      const res = await fetch("/status");
      const data = await res.json();

      let statusEl = document.getElementById("status");
      statusEl.innerText = data.label.toUpperCase();

      statusEl.className = "status-box";
      if (data.label === "real") {
        statusEl.classList.add("status-real");
      } else if (data.label === "fake") {
        statusEl.classList.add("status-fake");
      } else {
        statusEl.classList.add("status-waiting");
      }

      document.getElementById("confidence").innerText = "Confidence: " + data.conf + "%";
      document.getElementById("fps").innerText = "FPS: " + data.fps;
    }

    setInterval(updateStatus, 500); // update every 0.5s
  </script>
</body>
</html>
"""

# ----------------- VIDEO GENERATOR -----------------
def gen_frames():
    global prev_frame_time, current_label, current_conf, fps_value

    while True:
        success, img = cap.read()
        if not success:
            break

        new_frame_time = time.time()
        label = "waiting"
        conf_display = 0

        # Run YOLO detection
        results = model(img, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > confidence:
                    label = classNames[cls]
                    conf_display = int(conf * 100)
                    color = (0, 255, 0) if label == "real" else (0, 0, 255)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{label.upper()} {conf_display}%',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                       colorR=color, colorB=color)

        # FPS calculation
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time

        # Save status for JSON API
        current_label = label
        current_conf = conf_display
        fps_value = int(fps)

        # Draw FPS on video
        cvzone.putTextRect(img, f'FPS: {fps_value}', (20, 50), scale=2, thickness=2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return render_template_string(html_page)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        "label": current_label,
        "conf": current_conf,
        "fps": fps_value
    }

# ----------------- MAIN -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
