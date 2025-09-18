# Vchan

VChan is a lightweight FFmpeg-powered video channel streamer with HLS output, electronic program guide (EPG) support, and simple scheduling for local media playback.

---

## ✨ Features
- 🎬 Stream local video files as an **HLS channel**
- ⚡ Fast startup with optional hardware acceleration
- 📺 **EPG generation** (24h blocks, auto-refresh every 6h)
- 🕒 Scheduling system for continuous playback
- 🌐 Simple HTTP endpoints for health checks and playback
- 📝 Configurable via `.ini` or `.yaml`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH
- `pip install -r requirements.txt`

python vchan.py 8088
This will start the server on port 8088, serving HLS at:

http://localhost:8088/hls/channel.m3u8
Health check endpoint:
http://localhost:8088/health
