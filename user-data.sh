#!/bin/bash
set -eu

# -------- EDIT SESUAI PROYEK KAMU ----------------------------------------
GIT_REPO="https://github.com/<username>/<repo-nama-aplikasimu>.git"
SUBFOLDER=""                  
APP_FILE="app_cloud.py"             
ENDPOINT_NAME="credit-score-endpoint-5" 
# -------------------------------------------------------------------------

REGION="us-east-1"
APP_DIR="/home/ec2-user/my-app" 
VENV_DIR="/opt/streamlit-venv"

# Menentukan path aplikasi
if [ -z "$SUBFOLDER" ]; then
  APP_PATH="$APP_DIR"
else
  APP_PATH="$APP_DIR/$SUBFOLDER"
fi

# Update sistem dan instal dependencies
dnf update -y
dnf install -y python3 python3-pip git

# Clone repo
git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

# Instal virtual environment dan library dari requirements.txt
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
# Memastikan requirements.txt terinstal
if [ -f "$APP_PATH/requirements.txt" ]; then
    "$VENV_DIR/bin/pip" install -r "$APP_PATH/requirements.txt"
else
    "$VENV_DIR/bin/pip" install streamlit boto3 pandas
fi

# Membuat systemd service agar aplikasi jalan otomatis
cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service

# Logging status
echo "Streamlit service deployed successfully."