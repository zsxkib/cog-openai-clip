# Configuration for Cog ⚙️
# Reference: https://cog.run/yaml

build:
  gpu: true
  python_version: "3.11"
  python_requirements: "requirements.txt"
  system_packages:
    - "curl"
  run:
    - "curl -o /usr/local/bin/pget -L 'https://github.com/replicate/pget/releases/latest/download/pget_linux_x86_64' && chmod +x /usr/local/bin/pget"
predict: "predict.py:$PREDICTOR"
concurrency:
  max: 16
