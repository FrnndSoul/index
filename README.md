Index (Flask sensor API for Raspberry Pi)

Lightweight Flask API that exposes Raspberry Pi sensors and utilities through simple REST endpoints.
Each sensor lives in its own module under sensor-scripts/, and index.py acts as a dispatcher so routes stay clean.

Features

Generic dispatcher: GET/POST /api/<sensor>/<op>?params...

Ready examples (e.g., PIR):

GET /api/pir/status – health/status

GET /api/pir – read current value

GET /api/pir/config – read config

POST /api/pir/config – update config (JSON body)

Optional email alerts via resources/email_notifier.py

Optional camera helpers (Picamera2 / OpenCV), DHT/gas/vibration DB logging

1) Requirements
Raspberry Pi OS packages (once)
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev \
    libatlas-base-dev libffi-dev libssl-dev \
    libcamera-apps python3-picamera2 \
    git


If you’ll use OpenCV windows/video:

sudo apt install -y libgtk-3-0 libgl1 ffmpeg

Python packages

The project is light; the hard requirements are:

Flask

Flask-Cors

Optional (enable camera/vision or extras):

numpy, opencv-contrib-python (for CV helpers)

Any sensor libs you actually use (e.g., Adafruit_DHT, smbus2, etc.)

Install (system-wide user site on Pi; avoids venv, works with PEP 668):

python3 -m pip install --upgrade pip
python3 -m pip install --user --break-system-packages \
    Flask Flask-Cors
# Optional vision stack (only if you need it)
python3 -m pip install --user --break-system-packages \
    "numpy==1.26.4" "opencv-contrib-python==4.9.0.80" --no-deps

2) Clone & run
cd ~/Documents
git clone https://github.com/FrnndSoul/index.git
cd index
python3 index.py


By default the server listens on 0.0.0.0:5000.
Open a browser: http://<PI-IP>:5000/

3) API overview
Generic pattern
/api/<sensor>/<op>?param1=...&param2=...


index.py maps <sensor> to a module in sensor-scripts/ (via an alias table).

The module exposes a callable named api_<op>(ctx, **params) (or (ctx, params); the dispatcher handles both styles).

Examples (PIR)
GET  /api/pir/status
GET  /api/pir
GET  /api/pir/config
POST /api/pir/config       (JSON body)


Sample requests

# Status
curl "http://<PI-IP>:5000/api/pir/status"

# Read value
curl "http://<PI-IP>:5000/api/pir"

# Read config
curl "http://<PI-IP>:5000/api/pir/config"

# Update config (example keys depend on your module)
curl -X POST "http://<PI-IP>:5000/api/pir/config" \
     -H "Content-Type: application/json" \
     -d '{"enabled": true, "threshold": 0.5}'

Conventions (for writing handlers)

Inside sensor-scripts/<name>.py:

def api_status(ctx, **params): ...
def api_pir(ctx, **params):      # default read
def api_config(ctx, **params):   # GET config
def api_config_post(ctx, **params):  # if you prefer separate POST


Return plain dicts; the dispatcher jsonify()’s them.

4) Directory layout
index/
├─ index.py                  # Flask app + dispatcher
├─ test.py                   # small local tests (optional)
├─ sensor-scripts/
│  ├─ pir.py                 # PIR handlers (example)
│  ├─ dht.py                 # DHT handlers (if present)
│  ├─ gas_vib.py             # Gas/Vibration handlers (if present)
│  └─ ...                    # Add your sensors here
└─ resources/
   ├─ email_notifier.py      # Email alerts & cooldown logic
   ├─ dht_readings.db        # SQLite (if used)
   ├─ gv_data.db             # SQLite (if used)
   └─ ...                    # other assets/config


The dispatcher typically builds a CTX dict with paths/pins shared by modules.

5) Configuration

Edit constants in index.py:

Pins / Paths under CTX["pins"] and CTX["paths"]

Email settings in resources/email_notifier.py (sender, SMTP, cooldown)

If a sensor needs calibration files or secrets, put them in resources/ and load from there.

6) Running as a service (optional)

Create a systemd unit so it starts on boot:

sudo tee /etc/systemd/system/index.service >/dev/null <<'UNIT'
[Unit]
Description=Index Flask sensor API
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Documents/index
ExecStart=/usr/bin/python3 index.py
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now index.service
systemctl status index.service

7) Troubleshooting

PEP 668 / “externally managed environment”
Always include --user --break-system-packages when installing via pip on Raspberry Pi OS Bookworm.

Port already in use
Something else is using :5000. Change the port in index.py (e.g., app.run(host="0.0.0.0", port=5050)), or stop the other process.

CORS / browser fetch blocked
Ensure Flask-Cors is installed and enabled in index.py (CORS(app)).

Pi Camera access
Install Picamera2 (python3-picamera2) and run your camera endpoints using Picamera2 to produce NumPy frames for CV.

GPIO permission
Run as pi (not root) and ensure your sensor libraries are installed for Python 3.
For I2C/SPI sensors, enable the interfaces in raspi-config and add pi to needed groups.

8) Extend with a new sensor

Create sensor-scripts/my_sensor.py.

Implement handlers:

def api_status(ctx, **params): return {"ok": True}
def api_read(ctx, **params):   return {"ok": True, "value": 123}


Add an alias in index.py (if the project uses an _ALIASES dict), e.g.:

_ALIASES = {"my": "my_sensor"}


Call: GET /api/my/status, GET /api/my/read

9) Security note

This server is intended for LAN use. If exposing outside your network, put it behind a reverse proxy (nginx) with auth/TLS and validate inputs on mutating endpoints.

License

See repository license (if present). Otherwise, treat as educational/sample code for coursework.