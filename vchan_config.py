# vchan_config.py — Virtual Channel v0.7.1 (segment-safe channel.m3u)
# - Fix: /channel.m3u blocks briefly (up to WAIT_FOR_SEGMENTS_SEC) until
#        HLS manifest contains at least one #EXTINF line, preventing
#        "playlist had no segments" errors in picky transcoders/players.
# - Everything else retains v0.7 behavior and routes.
#
# Usage:
#   python vchan_config.py 8088 --profile cpu_x264
#
# Notes:
#   - M3U (IPTV):   http://<ip>:8088/channel.m3u
#   - EPG (XMLTV):  http://<ip>:8088/epg.xml
#   - HLS (VLC):    http://<ip>:8088/hls/channel.m3u8
# ------------------------------------------------------------------------------

import os, sys, json, time, html, atexit, signal, random, traceback, subprocess, threading, logging, logging.handlers, shlex
from pathlib import Path
from flask import Flask, Response, request, send_file, send_from_directory
import argparse, configparser

VERSION = "v0.7.2"

# ------------------------- config helpers -------------------------
def _try_load_yaml(path: Path):
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {})
    except Exception:
        return {}

def _as_list(val, sep=","):
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val]
    if isinstance(val, str):
        return [x.strip() for x in val.split(sep) if x.strip()]
    return []

def load_config(config_path: Path | None):
    base = Path(__file__).resolve().parent
    pl_dir = base / "playlist"
    pl_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_path) if config_path else (pl_dir / "vchan.ini")
    cfg = {}
    if cfg_path.suffix.lower() in (".yaml",".yml"):
        data = _try_load_yaml(cfg_path) or {}
        cfg.update({"format":"yaml", "yaml":data, "path":str(cfg_path)})
    else:
        cp = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        cp.read(cfg_path, encoding="utf-8")
        cfg.update({"format":"ini", "ini":cp, "path":str(cfg_path)})
    return cfg

def cfg_get(cfg, section, key, default=None):
    if cfg.get("format") == "yaml":
        return cfg.get("yaml", {}).get(section, {}).get(key, default)
    cp: configparser.ConfigParser = cfg.get("ini")
    if cp and cp.has_section(section) and cp.has_option(section, key):
        return cp.get(section, key)
    return default

def cfg_getint(cfg, section, key, default=None):
    v = cfg_get(cfg, section, key, None)
    if v is None: return default
    try: return int(float(v))
    except Exception: return default

def cfg_getfloat(cfg, section, key, default=None):
    v = cfg_get(cfg, section, key, None)
    if v is None: return default
    try: return float(v)
    except Exception: return default

def cfg_getlist(cfg, section, key, default=None):
    v = cfg_get(cfg, section, key, None)
    if v is None: return default or []
    return _as_list(v)

# ------------------------- paths & config -------------------------
BASE           = Path(__file__).resolve().parent
PL_DIR         = BASE / "playlist"
HLS_DIR        = BASE / "hls"
APP_LOG        = PL_DIR / "app.log"
FFLOG_FILE     = PL_DIR / "ffmpeg.log"
EPG_FILE       = PL_DIR / "epg.xml"
PID_FILE       = PL_DIR / "ffmpeg.pid"
STATE_FILE     = PL_DIR / "state.json"
ERROR_FILE     = PL_DIR / "last_error.txt"
SCHEDULE_FILE  = PL_DIR / "schedule.json"

_cli = argparse.ArgumentParser()
_cli.add_argument("port", nargs="?", type=int, default=None)
_cli.add_argument("--config", type=str, default=None)
_cli.add_argument("--profile", type=str, default=None)
args, _ = _cli.parse_known_args()

CFG = load_config(Path(args.config) if args.config else None)
# --- ENV overrides (container-friendly) ---
ENV = os.environ
def env_or(conf_val, key):
    v = ENV.get(key)
    return v if v is not None and v != "" else conf_val

SERVER_PORT    = args.port or int(env_or(cfg_getint(CFG, 'server', 'port', 8088), 'VCHAN_PORT'))
IDLE_TIMEOUT   = int(env_or(cfg_getint(CFG, 'server', 'idle_timeout_sec', 300), 'VCHAN_IDLE_TIMEOUT'))
SELECTED_PROFILE = (args.profile or env_or(cfg_get(CFG, 'server', 'selected_profile', 'cpu_x264'), 'VCHAN_PROFILE'))

LOG_MAX_BYTES  = cfg_getint(CFG, "logging", "max_bytes", 10*1024*1024)
LOG_BACKUPS    = cfg_getint(CFG, "logging", "backups", 3)

EPG_REFRESH_S  = cfg_getint(CFG, "epg", "refresh_seconds", 6*3600)
RESCAN_EVERY_S = cfg_getint(CFG, "scan", "rescan_seconds", 5*60)

MEDIA_DIR      = Path(env_or(cfg_get(CFG, 'media', 'media_dir', r'C:\vchan\media'), 'VCHAN_MEDIA_DIR'))
ALLOWED_EXTS   = set(x.lower() for x in cfg_getlist(CFG, "media", "allowed_exts", [".mp4",".mkv",".mov",".m4v",".mpg",".ts",".m2ts"]))
MIN_PROG_SEC   = cfg_getint(CFG, "media", "min_programme_seconds", 30*60)

EPG_CHANNEL_ID   = cfg_get(CFG, "channel", "id", "VChannel.1")
EPG_CHANNEL_NAME = cfg_get(CFG, "channel", "name", "My Virtual Channel")

HLS_SUBDIR      = env_or(cfg_get(CFG, 'hls', 'dir', 'hls'), 'VCHAN_HLS_DIR')
HLS_DIR          = BASE / HLS_SUBDIR
HLS_MANIFEST     = env_or(cfg_get(CFG, 'hls', 'manifest', 'channel.m3u8'), 'VCHAN_HLS_MANIFEST')
HLS_TIME         = int(env_or(cfg_getint(CFG, 'hls', 'time', 4), 'VCHAN_HLS_TIME'))
HLS_LIST_SIZE    = int(env_or(cfg_getint(CFG, 'hls', 'list_size', 30), 'VCHAN_HLS_LIST_SIZE'))
HLS_FLAGS        = env_or(cfg_get(CFG, 'hls', 'flags', 'append_list+program_date_time+independent_segments+discont_start'), 'VCHAN_HLS_FLAGS')
HLS_CLEAN_SECONDS= int(env_or(cfg_getint(CFG, 'hls', 'cleanup_keep_seconds', 1200), 'VCHAN_HLS_CLEAN_SECONDS'))

# NEW: allow tuning how long /channel.m3u waits for first segment
WAIT_FOR_SEGMENTS_SEC = cfg_getint(CFG, "server", "wait_for_segments_sec", 5)

HLS_DIR.mkdir(parents=True, exist_ok=True); PL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- app & logging -------------------------
app = Flask(__name__, static_folder=None)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

def _rotate(path: Path):
    try:
        if not path.exists() or path.stat().st_size < LOG_MAX_BYTES: return
        for i in range(LOG_BACKUPS, 0, -1):
            src = path if i == 1 else path.with_suffix(path.suffix + f".{i-1}")
            dst = path.with_suffix(path.suffix + f".{i}")
            if src.exists():
                if dst.exists():
                    try: dst.unlink()
                    except Exception: pass
                try: src.rename(dst)
                except Exception: pass
        path.touch()
    except Exception:
        pass

def log_event(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{VERSION}] {msg}\n"
    try:
        _rotate(APP_LOG)
        with open(APP_LOG, "ab") as f:
            f.write(line.encode("utf-8", errors="ignore"))
    except Exception:
        pass

def note_error(msg: str):
    log_event(f"ERROR: {msg}")
    try: ERROR_FILE.write_text(msg, encoding="utf-8")
    except Exception: pass

# uncaught hooks
def _uncaught(exctype, value, tb):
    txt = "UNCAUGHT EXCEPTION: " + "".join(traceback.format_exception(exctype, value, tb))
    note_error(txt)
import traceback
sys.excepthook = _uncaught

# ---------------------- state/threads -------------------------
SUPERVISOR_ACTIVE   = threading.Event()
PROCESS_SHOULD_EXIT = threading.Event()
manifest_created_once = threading.Event()

_supervisor_thread = None
_watchdog_thread   = None
_rescan_thread     = None
_epg_thread        = None
_flask_thread      = None
_hls_pruner_thread = None

_last_access_ts   = 0.0
_last_access_lock = threading.Lock()

_media_cache = []
_media_lock  = threading.Lock()
_last_scan   = 0.0

def touch_access():
    global _last_access_ts
    with _last_access_lock:
        _last_access_ts = time.time()

def seconds_since_last_access() -> float:
    with _last_access_lock:
        return 1e9 if _last_access_ts == 0 else (time.time() - _last_access_ts)

# -------------------------- ffprobe utils ----------------------------
def ffprobe_json(args) -> dict:
    out = subprocess.check_output(["ffprobe","-v","error","-of","json", *args], stderr=subprocess.STDOUT)
    return json.loads(out or b"{}")

def ffprobe_duration(path: Path) -> float:
    try:
        data = subprocess.check_output(["ffprobe","-v","error","-of","json","-show_entries","format=duration", str(path)], stderr=subprocess.STDOUT)
        j = json.loads(data)
        return max(float(j["format"]["duration"]), 0.0)
    except Exception:
        return 0.0

def ffprobe_audio_codec(path: Path) -> str:
    try:
        data = subprocess.check_output(["ffprobe","-v","error","-of","json","-show_streams","-select_streams","a:0", str(path)], stderr=subprocess.STDOUT)
        j = json.loads(data); streams = j.get("streams", [])
        if not streams: return ""
        return (streams[0].get("codec_name") or "").lower()
    except Exception:
        return ""

# -------------------------- media scanning ---------------------------
def list_media_now() -> list[Path]:
    return [p for p in MEDIA_DIR.rglob("*") if p.suffix.lower() in ALLOWED_EXTS]

def get_media_list() -> list[Path]:
    with _media_lock:
        return list(_media_cache)

def rescan_media_loop():
    global _media_cache, _last_scan
    log_event("Media rescan thread starting")
    while not PROCESS_SHOULD_EXIT.is_set():
        try:
            files = list_media_now()
            with _media_lock:
                _media_cache = files
                _last_scan = time.time()
            log_event(f"Media rescan: found {len(files)} items")
        except Exception as e:
            note_error(f"Media rescan failed: {e}")
        if PROCESS_SHOULD_EXIT.wait(RESCAN_EVERY_S):
            break
    log_event("Media rescan thread stopped")

def ensure_rescanner():
    global _rescan_thread, _media_cache, _last_scan
    if _rescan_thread and _rescan_thread.is_alive(): return
    try:
        files = list_media_now()
        with _media_lock:
            _media_cache = files
            _last_scan = time.time()
        log_event(f"Initial media scan: {len(files)} items")
    except Exception as e:
        note_error(f"Initial media scan failed: {e}")
    _rescan_thread = threading.Thread(target=rescan_media_loop, name="vchan-rescan", daemon=False)
    _rescan_thread.start()

# ------------------------ Schedule (24h) -----------------------
def pretty_title_from_path(p: Path) -> str:
    return " ".join(p.stem.replace("_"," ").replace("."," ").split())

def xmltv_dt(ts: float) -> str:
    return time.strftime("%Y%m%d%H%M%S %z", time.localtime(ts))

def build_schedule(files: list[Path], start_ts: float | None = None) -> list[dict]:
    if not files: return []
    now = time.time()
    if start_ts is None:
        start_ts = now - (now % 60)
    rnd = random.Random(int(start_ts) // max(EPG_REFRESH_S, 1))
    order = files[:]; rnd.shuffle(order)
    t = start_ts; end = start_ts + 24*3600; i = 0
    sched = []
    while t < end and order:
        f = order[i % len(order)]; i += 1
        dur = ffprobe_duration(f) or float(MIN_PROG_SEC)
        start = t; stop = min(t + dur, end)
        sched.append({"file": str(f), "start": int(start), "stop": int(stop), "title": pretty_title_from_path(f), "duration": int(stop-start)})
        t = stop
    return sched

def save_schedule(sched: list[dict]):
    try: SCHEDULE_FILE.write_text(json.dumps(sched, indent=2), encoding="utf-8")
    except Exception as e: note_error(f"save_schedule error: {e}")

def load_schedule() -> list[dict]:
    if not SCHEDULE_FILE.exists(): return []
    try: return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except Exception as e: note_error(f"load_schedule error: {e}"); return []

def ensure_schedule() -> list[dict]:
    files = get_media_list()
    if not files: return []
    sched = load_schedule()
    now = int(time.time())
    if not sched or sched[-1]["stop"] <= now:
        log_event("Building new schedule (24h)")
        sched = build_schedule(files, start_ts=now)
        save_schedule(sched)
    return sched

def get_current_programme(now: float | None = None) -> dict | None:
    sched = ensure_schedule()
    if not sched: return None
    t = time.time() if now is None else now
    for p in sched:
        if p["start"] <= t < p["stop"]:
            return p
    return None

# ------------------------ EPG ----------------------------
def build_epg_from_schedule(sched: list[dict]) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<tv generator-info-name="vchan" source-info-name="local">', f'  <channel id="{EPG_CHANNEL_ID}"><display-name>{html.escape(EPG_CHANNEL_NAME)}</display-name></channel>']
    for p in sched:
        out.append(f'  <programme start="{xmltv_dt(p["start"])}" stop="{xmltv_dt(p["stop"])}" channel="{EPG_CHANNEL_ID}">')
        out.append(f'    <title>{html.escape(p["title"])}</title>')
        out.append('  </programme>')
    out.append('</tv>'); return "\n".join(out)

def epg_loop():
    log_event("EPG thread starting")
    while not PROCESS_SHOULD_EXIT.is_set():
        try:
            sched = ensure_schedule()
            EPG_FILE.write_text(build_epg_from_schedule(sched), encoding="utf-8")
            log_event("EPG regenerated")
        except Exception as e:
            note_error(f"EPG build failed: {e}")
        if PROCESS_SHOULD_EXIT.wait(EPG_REFRESH_S): break
    log_event("EPG thread stopped")

def ensure_epg_thread():
    global _epg_thread
    if _epg_thread and _epg_thread.is_alive(): return
    _epg_thread = threading.Thread(target=epg_loop, name="vchan-epg", daemon=False); _epg_thread.start()

# ------------------- FFmpeg command builder --------------------------
def ff_vars_extra():
    fps = int(env_or(cfg_getfloat(CFG, "ffmpeg.global", "fps", 30) or 30, "VCHAN_FPS"))
    gops = int(env_or(cfg_getfloat(CFG, "ffmpeg.global", "gop_seconds", 2) or 2, "VCHAN_GOP_SECONDS"))
    return {
        "width":          env_or(cfg_get(CFG, "ffmpeg.global", "width", "1280"), "VCHAN_WIDTH"),
        "height":         env_or(cfg_get(CFG, "ffmpeg.global", "height", "720"), "VCHAN_HEIGHT"),
        "fps":            str(fps),
        "pix_fmt":        env_or(cfg_get(CFG, "ffmpeg.global", "pix_fmt", "yuv420p"), "VCHAN_PIX_FMT"),
        "video_bitrate":  env_or(cfg_get(CFG, "ffmpeg.global", "video_bitrate", "5M"), "VCHAN_VIDEO_BITRATE"),
        "audio_bitrate":  env_or(cfg_get(CFG, "ffmpeg.global", "audio_bitrate", "128k"), "VCHAN_AUDIO_BITRATE"),
        "g":              str(int(fps) * int(gops)),
        "hls_time":       str(HLS_TIME),
        "hls_list_size":  str(HLS_LIST_SIZE),
        "hls_flags":      HLS_FLAGS,
        "hls_dir":        str(HLS_DIR),
        "manifest_path":  str(HLS_DIR / HLS_MANIFEST),
        "media_dir":      str(MEDIA_DIR),
        "version":        VERSION,
    }

def build_ffmpeg_cmd(src: Path, start_at: float, ensure_aac: bool) -> list[str]:
    vm = ff_vars_extra()
    vm.update({"input": str(src), "output": str(HLS_DIR / HLS_MANIFEST), "start_at": f"{start_at:.3f}"})
    # audio policy: copy if aac else transcode to aac
    acodec = ffprobe_audio_codec(src)
    if ensure_aac and acodec != "aac":
        vm["audio_args"] = f"-c:a aac -ac 2 -b:a {vm['audio_bitrate']}"
    else:
        vm["audio_args"] = "-c:a copy"

    # resolve profile template
    if CFG.get("format") == "yaml":
        node = CFG.get("yaml", {}).get("ffmpeg", {})
        g = node.get("global", {}); p = node.get("profiles", {}).get(SELECTED_PROFILE, {})
        tmpl = p.get("cmd") or ""
        for k,v in g.items(): vm.setdefault(k, str(v))
        for k,v in p.items():
            if k != "cmd": vm[k] = str(v)
    else:
        cp: configparser.ConfigParser = CFG.get("ini")
        g = dict(cp.items("ffmpeg.global")) if cp.has_section("ffmpeg.global") else {}
        sect = f"ffmpeg.profile.{SELECTED_PROFILE}"
        p = dict(cp.items(sect)) if cp.has_section(sect) else {}
        tmpl = p.get("cmd", "")
        for k,v in g.items(): vm.setdefault(k, str(v))
        for k,v in p.items():
            if k != "cmd": vm[k] = str(v)
    if not tmpl: raise RuntimeError(f"Profile '{SELECTED_PROFILE}' has no cmd")
    args_str = f"ffmpeg -hide_banner -nostdin {tmpl.format(**vm)}"
    return shlex.split(args_str)

# ------------------- process & supervisor --------------------------
def running_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip()); os.kill(pid, 0); return pid
        except Exception:
            try: PID_FILE.unlink()
            except Exception: pass
    return None

def write_state(d: dict):
    d["version"] = VERSION; d["profile"] = SELECTED_PROFILE
    try: STATE_FILE.write_text(json.dumps(d, indent=2))
    except Exception: pass

def _popen_group_kwargs():
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"preexec_fn": os.setsid}

def create_initial_manifest_once():
    # Keep starter manifest; /channel.m3u waits for #EXTINF before returning it.
    if manifest_created_once.is_set(): return
    try:
        initial = f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:{HLS_TIME}\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-DISCONTINUITY\n"
        (HLS_DIR / HLS_MANIFEST).write_text(initial, encoding="utf-8")
        manifest_created_once.set()
    except Exception as e:
        log_event(f"Initial manifest failed: {e}")

def launch_ffmpeg_for_file(src: Path, start_at: float, ensure_aac: bool) -> subprocess.Popen:
    try: dur = ffprobe_duration(src)
    except Exception: dur = 0.0
    if start_at >= max(0.0, dur): start_at = max(0.0, dur - 5.0)
    ac = ffprobe_audio_codec(src)
    write_state({"started_at": int(time.time()), "file": str(src), "start_offset": float(start_at), "audio_codec": ac})
    log_event(f"FFmpeg start profile={SELECTED_PROFILE} file={src.name} start_at={start_at:.2f}s audio={ac or 'n/a'}")
    try: open(FFLOG_FILE, "ab").close()
    except Exception: pass
    cmd = build_ffmpeg_cmd(src, start_at, ensure_aac=True)
    fflog = open(FFLOG_FILE, "ab")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=fflog, **_popen_group_kwargs())
    PID_FILE.write_text(str(proc.pid))
    return proc

def _terminate_by_pid(pid: int, graceful_timeout=3):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill","/PID",str(pid),"/T","/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                pgid = os.getpgid(pid); os.killpg(pgid, signal.SIGTERM)
            except Exception:
                try: os.kill(pid, signal.SIGTERM)
                except Exception: pass
            deadline = time.time() + graceful_timeout
            while time.time() < deadline:
                try: os.kill(pid, 0); time.sleep(0.2)
                except ProcessLookupError: return
            try: pg = os.getpgid(pid); os.killpg(pg, signal.SIGKILL)
            except Exception:
                try: os.kill(pid, signal.SIGKILL)
                except Exception: pass
    except Exception as e:
        log_event(f"_terminate_by_pid error: {e}")

def stop_ffmpeg_only():
    pid = running_pid()
    if pid:
        try: log_event(f"Stopping FFmpeg pid {pid}"); _terminate_by_pid(pid)
        except Exception as e:
            log_event(f"Error stopping FFmpeg: {e}")
        try: PID_FILE.unlink()
        except Exception: pass

def supervisor_loop():
    log_event("Supervisor loop starting")
    crash_times = []
    try:
        while not PROCESS_SHOULD_EXIT.is_set():
            if not SUPERVISOR_ACTIVE.is_set():
                PROCESS_SHOULD_EXIT.wait(5); continue
            files = get_media_list()
            if not files:
                log_event(f"No media files in {MEDIA_DIR}")
                if PROCESS_SHOULD_EXIT.wait(5): break
                continue
            prog = get_current_programme()
            if not prog:
                log_event("No programme scheduled")
                if PROCESS_SHOULD_EXIT.wait(3): break
                continue
            current = Path(prog["file"])
            start_at = max(0.0, time.time() - prog["start"])
            create_initial_manifest_once()
            proc = launch_ffmpeg_for_file(current, start_at, ensure_aac=True)

            def m3u_ok():
                try:
                    m3u = (HLS_DIR / HLS_MANIFEST)
                    if not m3u.exists() or m3u.stat().st_size == 0: return False
                    txt = m3u.read_text(errors="ignore")
                    # consider manifest "ok" only once we see segments
                    return "#EXTINF" in txt
                except Exception: return False

            for i in range(50):
                if not SUPERVISOR_ACTIVE.is_set() or PROCESS_SHOULD_EXIT.is_set(): break
                if m3u_ok():
                    if i: log_event(f"Manifest ready after {i*0.1:.1f}s")
                    break
                if proc.poll() is not None:
                    note_error("ffmpeg exited during startup; see /logs/ffmpeg")
                    break
                time.sleep(0.1)

            TICK = 0.2; last_check = 0.0
            while SUPERVISOR_ACTIVE.is_set() and not PROCESS_SHOULD_EXIT.is_set():
                ret = proc.poll()
                if ret is not None:
                    log_event(f"FFmpeg exited with code {ret}")
                    now = time.time()
                    crash_times = [t for t in crash_times if now - t < 15] + [now]
                    if len(crash_times) >= 3:
                        log_event("FFmpeg failing repeatedly, backoff 10s"); time.sleep(10)
                    break

                now = time.time()
                if now - last_check >= 1.0:
                    last_check = now
                    prog2 = get_current_programme(now)
                    if not prog2:
                        log_event("No programme scheduled now; stopping")
                        _terminate_by_pid(proc.pid); break
                    if Path(prog2["file"]) != current:
                        log_event(f"Programme advance -> {Path(prog2['file']).name}")
                        _terminate_by_pid(proc.pid)
                        current = Path(prog2["file"])
                        new_start = max(0.0, time.time() - prog2["start"])
                        proc = launch_ffmpeg_for_file(current, new_start, ensure_aac=True)
                        time.sleep(0.25)
                    elif now >= prog2["stop"]:
                        log_event(f"Programme end: {prog2['title']}")
                        _terminate_by_pid(proc.pid); break
                time.sleep(TICK)
    except Exception as e:
        if not PROCESS_SHOULD_EXIT.is_set():
            note_error(f"Supervisor error: {e}")
    finally:
        log_event("Supervisor loop ending")

def ensure_supervisor_running():
    global _supervisor_thread
    if _supervisor_thread and _supervisor_thread.is_alive(): return
    _supervisor_thread = threading.Thread(target=supervisor_loop, name="vchan-supervisor", daemon=False)
    _supervisor_thread.start(); log_event("Supervisor started")

def stop_supervisor():
    log_event("stop_supervisor(): begin")
    SUPERVISOR_ACTIVE.clear()
    stop_ffmpeg_only()
    log_event("stop_supervisor(): complete")

# ----------------------- idle watchdog ---------------------------------
def watchdog_loop():
    log_event("Watchdog thread starting")
    while not PROCESS_SHOULD_EXIT.is_set():
        PROCESS_SHOULD_EXIT.wait(5)
        if SUPERVISOR_ACTIVE.is_set() and seconds_since_last_access() > IDLE_TIMEOUT:
            elapsed = seconds_since_last_access()
            log_event(f"Idle timeout ({elapsed:.1f}s > {IDLE_TIMEOUT}s) - stopping")
            SUPERVISOR_ACTIVE.clear(); stop_ffmpeg_only()
    log_event("Watchdog thread stopped")

def ensure_watchdog_running():
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive(): return
    _watchdog_thread = threading.Thread(target=watchdog_loop, name="vchan-watchdog", daemon=False)
    _watchdog_thread.start(); log_event("Watchdog started")

# ----------------------- HLS PRUNER ---------------------------------
def hls_prune_loop():
    log_event("HLS pruner starting")
    while not PROCESS_SHOULD_EXIT.is_set():
        try:
            if HLS_CLEAN_SECONDS and HLS_CLEAN_SECONDS > 0:
                now = time.time()
                for p in HLS_DIR.glob("*.ts"):
                    try:
                        if now - p.stat().st_mtime > max(60, HLS_CLEAN_SECONDS):
                            p.unlink()
                    except Exception: pass
        except Exception as e:
            log_event(f"HLS prune error: {e}")
        if PROCESS_SHOULD_EXIT.wait(60): break
    log_event("HLS pruner stopped")

def ensure_hls_pruner():
    global _hls_pruner_thread
    if _hls_pruner_thread and _hls_pruner_thread.is_alive(): return
    _hls_pruner_thread = threading.Thread(target=hls_prune_loop, name="vchan-hls-pruner", daemon=False)
    _hls_pruner_thread.start(); log_event("HLS pruner started")

# --------------------------- HTTP ------------------------------------
def _client_str():
    ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
    ua = request.headers.get("User-Agent","")
    return f"{ip} UA='{ua[:140]}'"

@app.route("/channel.m3u")
def bootstrap_m3u():
    touch_access(); log_event(f"/channel.m3u requested by {_client_str()}")
    try:
        ensure_rescanner(); ensure_epg_thread(); ensure_watchdog_running(); ensure_hls_pruner(); ensure_supervisor_running(); SUPERVISOR_ACTIVE.set()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"; note_error(msg)
        return Response("#EXTM3U\n#EXTINF:-1,Error starting channel\n# "+msg+"\n", mimetype="audio/x-mpegurl", status=500)

    # NEW: Wait until the manifest contains at least one segment before returning.
    deadline = time.time() + max(0, int(WAIT_FOR_SEGMENTS_SEC or 0))
    man_path = (HLS_DIR / HLS_MANIFEST)
    waited = 0.0
    while time.time() < deadline:
        try:
            if man_path.exists() and man_path.stat().st_size > 0:
                txt = man_path.read_text(errors="ignore")
                if "#EXTINF" in txt:
                    break
        except Exception:
            pass
        time.sleep(0.1); waited += 0.1
    if waited:
        log_event(f"/channel.m3u waited {waited:.1f}s for first segment")

    host = f"{request.scheme}://{request.host}"
    m3u = ("#EXTM3U\n"
           f'#EXTINF:-1 tvg-id="{EPG_CHANNEL_ID}" tvg-name="{EPG_CHANNEL_NAME}" group-title="Virtual",{EPG_CHANNEL_NAME} ({VERSION})\n'
           f"{host}/hls/{HLS_MANIFEST}\n")
    resp = Response(m3u, mimetype="audio/x-mpegurl"); resp.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"; resp.headers["Pragma"]="no-cache"; return resp

@app.route("/hls/<path:filename>")
def serve_hls(filename):
    touch_access(); log_event(f"/hls/{filename} requested by {_client_str()}")
    if filename == HLS_MANIFEST and not SUPERVISOR_ACTIVE.is_set():
        log_event("Bootstrap supervisor on manifest request")
        try:
            ensure_rescanner(); ensure_epg_thread(); ensure_watchdog_running(); ensure_hls_pruner(); ensure_supervisor_running(); SUPERVISOR_ACTIVE.set()
        except Exception as e:
            note_error(f"hls bootstrap failed: {e}"); return Response(f"HLS bootstrap failed: {e}", mimetype="text/plain", status=503)
    p = HLS_DIR / filename
    if not p.exists():
        log_event(f"File not found: {p}"); return Response(f"File not found: {filename}", mimetype="text/plain", status=404)
    try:
        resp = send_from_directory(str(HLS_DIR), filename, conditional=True)
        resp.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"; resp.headers["Pragma"]="no-cache"; return resp
    except Exception as e:
        log_event(f"Error serving {filename}: {e}"); return Response(f"Error serving file: {e}", mimetype="text/plain", status=500)

@app.route("/epg.xml")
def serve_epg():
    if EPG_FILE.exists(): return send_file(str(EPG_FILE), mimetype="application/xml")
    return Response("EPG not generated yet.", mimetype="text/plain", status=503)

@app.route("/health")
def health():
    pid = None
    if PID_FILE.exists():
        try: pid = int(PID_FILE.read_text().strip())
        except Exception: pid = None
    try: state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except Exception: state = {}
    try: err = ERROR_FILE.read_text() if ERROR_FILE.exists() else ""
    except Exception: err = ""
    manifest_exists = (HLS_DIR / HLS_MANIFEST).exists()
    now_prog = get_current_programme(); sched = load_schedule()
    return {
        "version": VERSION, "profile": SELECTED_PROFILE, "supervisor_active": SUPERVISOR_ACTIVE.is_set(),
        "running": bool(pid), "pid": pid, "manifest": manifest_exists,
        "idle_timeout_sec": IDLE_TIMEOUT, "seconds_since_last_access": round(seconds_since_last_access(), 1),
        "media_count": len(get_media_list()), "last_media_scan": int(_last_scan) if _last_scan else 0,
        "epg_last_modified": int(EPG_FILE.stat().st_mtime) if EPG_FILE.exists() else 0,
        "schedule_now": now_prog, "schedule_until": (sched[-1]["stop"] if sched else 0),
        "state": state, "last_error": err
    }

@app.route("/logs/app")
def logs_app():
    if APP_LOG.exists(): return send_file(str(APP_LOG), mimetype="text/plain")
    return Response("No app.log yet.", mimetype="text/plain")

@app.route("/logs/ffmpeg")
def logs_ffmpeg():
    if FFLOG_FILE.exists(): return send_file(str(FFLOG_FILE), mimetype="text/plain")
    return Response("No ffmpeg.log yet.", mimetype="text/plain")

@app.route("/stop", methods=["POST"])
def stop():
    stop_supervisor(); return {"stopped": True, "version": VERSION}

@app.route("/shutdown", methods=["POST"])
def shutdown():
    log_event("Shutdown requested via HTTP"); PROCESS_SHOULD_EXIT.set(); stop_supervisor(); return {"shutdown": True, "version": VERSION}

# --------------------------- Flask BG ----------------------
def run_flask_background(port: int):
    try:
        log_event("Flask background start")
        app.config['PROPAGATE_EXCEPTIONS'] = False
        app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False, debug=False)
    except Exception as e:
        log_event(f"Flask crashed: {e}"); note_error(f"Flask crashed: {e}")

def ensure_flask_running(port: int):
    global _flask_thread
    if _flask_thread and _flask_thread.is_alive(): return
    _flask_thread = threading.Thread(target=run_flask_background, args=(port,), daemon=False, name="vchan-flask")
    _flask_thread.start()

# --------------------------- main ----------------------
if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except Exception:
        pass

    log_event(f"=== VCHAN {VERSION} starting (profile={SELECTED_PROFILE}) ===")
    touch_access()
    ensure_rescanner(); ensure_epg_thread(); ensure_watchdog_running(); ensure_hls_pruner(); ensure_supervisor_running(); ensure_flask_running(SERVER_PORT)

    # Keep root alive & restart guards
    threads = [_rescan_thread, _epg_thread, _watchdog_thread, _hls_pruner_thread, _supervisor_thread, _flask_thread]
    beat = 0
    try:
        while True:
            if PROCESS_SHOULD_EXIT.wait(10): break
            beat += 1
            if beat % 6 == 0:
                alive = sum(1 for t in threads if t and t.is_alive())
                log_event(f"HEARTBEAT: active={SUPERVISOR_ACTIVE.is_set()} last_access={seconds_since_last_access():.1f}s threads_alive={alive}/{len([t for t in threads if t])}")
                if not _flask_thread or not _flask_thread.is_alive(): ensure_flask_running(SERVER_PORT)
                if not _supervisor_thread or not _supervisor_thread.is_alive(): ensure_supervisor_running()
                if not _watchdog_thread or not _watchdog_thread.is_alive(): ensure_watchdog_running()
                if not _hls_pruner_thread or not _hls_pruner_thread.is_alive(): ensure_hls_pruner()
    except KeyboardInterrupt:
        PROCESS_SHOULD_EXIT.set()
    log_event("Shutting down..."); stop_ffmpeg_only()
    for t in threads:
        if t and t.is_alive(): t.join(timeout=2)
    log_event("=== VCHAN shutdown complete ===")
