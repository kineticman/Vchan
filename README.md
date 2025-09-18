# VChan

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

### Run
```bash
python vchan.py 8088
```

This will start the server on port **8088**, serving HLS at:
```
http://localhost:8088/hls/channel.m3u8
```

Health check endpoint:
```
http://localhost:8088/health
```

---

## ⚙️ Configuration
Edit `vchan_config.ini` (or YAML version) to set:
- Playback mode (CPU/GPU, codec options)
- Idle timeout
- Logging options
- EPG refresh interval
- Media folder path

Example:
```ini
[general]
idle_timeout_sec = 60
media_path = C:\vchan\media

[ffmpeg]
codec = h264
preset = veryfast
```

---

## 📂 Project Structure
```
VChan/
├── vchan.py              # main service
├── vchan_config.ini      # configuration
├── /media                # local video files
├── /hls                  # generated HLS output
└── README.md
```

---

## 🔮 Roadmap
- [ ] Add live stream input support (e.g., RTMP, IPTV)
- [ ] Web UI for EPG & scheduling
- [ ] Docker container support
- [ ] Remote management API

---

## 🤝 Contributing
Pull requests welcome. For major changes, please open an issue first to discuss.

---

## 📜 License
MIT License

