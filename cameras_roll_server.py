#!/usr/bin/env python3
import argparse
import time
from glob import glob
from pathlib import Path
import signal
import sys
from dataclasses import dataclass
import threading
import socket
import socketserver
import json
from typing import Tuple, Optional, Dict, Set, Any, List
import os

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
from PIL import Image, ImageDraw, ImageFont

# ---------------- utils ----------------
def latest_best(weights_glob="best.pt", fallback="yolo11n.pt"):
    paths = [Path(p) for p in glob(weights_glob)]
    if not paths:
        print(f"[WARN] No se encontraron pesos en {weights_glob}. Uso fallback: {fallback}")
        return fallback
    best = max(paths, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Usando pesos más recientes: {best}")
    return str(best)

def fmt_ts(seconds: float) -> str:
    if seconds is None:
        return "--:--:--.---"
    ms = int(round((seconds - int(seconds)) * 1000.0))
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def interp_cross_ts(prev_cx, cx, thresh_x, prev_frame_idx, in_fps):
    if cx == prev_cx:
        alpha = 1.0
    else:
        alpha = (thresh_x - prev_cx) / (cx - prev_cx)
        alpha = np.clip(alpha, 0.0, 1.0)
    return (prev_frame_idx + alpha) / in_fps

def is_stream_url(src: str) -> bool:
    s = (src or "").lower()
    return s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"))

@dataclass
class TrackState:
    id: int
    prev_cx: float | None = None
    prev_cy: float | None = None
    first_seen_ts: float | None = None
    crossed_left_ts: float | None = None
    crossed_right_ts: float | None = None
    counted: bool = False
    last_seen_mono: float | None = None   # última vez visto (monotonic)

# ---------------- args ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Roll counter dual-cam con servidor de eventos + stream MJPEG + fusión.")
    # Fuentes
    p.add_argument("--source1", required=True, help="Ruta/URL Cam1 (archivo o IP: rtsp/http/rtmp/udp/tcp) – dirección por defecto R2L")
    p.add_argument("--source2", required=True, help="Ruta/URL Cam2 (archivo o IP: rtsp/http/rtmp/udp/tcp) – dirección por defecto L2R")
    p.add_argument("--dir1", default="R2L", choices=["L2R", "R2L"], help="Dirección de Cam1 (default R2L)")
    p.add_argument("--dir2", default="L2R", choices=["L2R", "R2L"], help="Dirección de Cam2 (default L2R)")

    # Modelo / tracking
    p.add_argument("--conf", type=float, default=0.30, help="Umbral confianza")
    p.add_argument("--img", type=int, default=768, help="Image size")
    p.add_argument("--device", default=None, help="GPU id (0) o 'cpu'")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="IDs de clases a usar (ej. 0)")
    p.add_argument("--weights_glob", default="best.pt", help="Patrón para best.pt")
    p.add_argument("--fallback_weights", default="yolo11n.pt", help="Fallback pesos")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (Ultralytics)")
    p.add_argument("--min_track_len", type=int, default=2, help="Frames mínimos antes de validar entrada")

    # Visualización / salida
    p.add_argument("--show", action="store_true", help="Mostrar ventana local (mosaico 2x1 + panel inferior)")
    # (quitado --save_video: ahora se maneja con botón en pantalla)
    p.add_argument("--out", default="", help="Ruta salida de videos (prefijo de carpeta)")
    p.add_argument("--fps_cap", type=float, default=0, help="FPS display/procesamiento (0=sin límite)")
    p.add_argument("--slowdown", type=float, default=1.0, help="Factor de ralentización visual/procesamiento (>=1.0)")
    p.add_argument("--active_timeout", type=float, default=2.0, help="Segundos sin ver un track para ocultarlo del panel y purgarlo")

    # Tipografía UI
    p.add_argument("--ui_font_path", type=str, default="",
                   help="Ruta a fuente TTF/OTF para el panel (ej. /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf)")
    p.add_argument("--ui_font_size", type=int, default=18,
                   help="Tamaño de fuente para el panel (px)")

    # Geometría global (defaults) y overrides por cámara
    p.add_argument("--left_ratio",  type=float, default=0.35, help="Barrera izquierda X [0-1] default")
    p.add_argument("--right_ratio", type=float, default=0.65, help="Barrera derecha X [0-1] default")
    p.add_argument("--band_top_ratio", type=float, default=0.35, help="Corredor top Y [0-1] default")
    p.add_argument("--band_bot_ratio", type=float, default=0.65, help="Corredor bottom Y [0-1] default")
    p.add_argument("--band_slack_px", type=int, default=16, help="Tolerancia vertical extra en px default")

    p.add_argument("--left_ratio1", type=float, default=None, help="Override Cam1 barrera izquierda X [0-1]")
    p.add_argument("--right_ratio1", type=float, default=None, help="Override Cam1 barrera derecha X [0-1]")
    p.add_argument("--band_top_ratio1", type=float, default=None, help="Override Cam1 corredor top Y [0-1]")
    p.add_argument("--band_bot_ratio1", type=float, default=None, help="Override Cam1 corredor bottom Y [0-1]")
    p.add_argument("--band_slack_px1", type=int, default=None, help="Override Cam1 tolerancia vertical extra en px")

    p.add_argument("--left_ratio2", type=float, default=None, help="Override Cam2 barrera izquierda X [0-1]")
    p.add_argument("--right_ratio2", type=float, default=None, help="Override Cam2 barrera derecha X [0-1]")
    p.add_argument("--band_top_ratio2", type=float, default=None, help="Override Cam2 corredor top Y [0-1]")
    p.add_argument("--band_bot_ratio2", type=float, default=None, help="Override Cam2 corredor bottom Y [0-1]")
    p.add_argument("--band_slack_px2", type=int, default=None, help="Override Cam2 tolerancia vertical extra en px")

    # Fusión (deduplicación)
    p.add_argument("--match_window", type=float, default=0.6, help="Δt máx (s) para matchear cam1-cam2 como mismo rollo")
    p.add_argument("--hold_window", type=float, default=1.2, help="Tiempo de espera (s) antes de emitir single-cam")

    # Servidores
    p.add_argument("--host", default="0.0.0.0", help="Host bind para ambos servidores")
    p.add_argument("--http_port", type=int, default=0, help="Puerto HTTP (0=random)")
    p.add_argument("--event_port", type=int, default=0, help="Puerto TCP eventos (0=random)")
    return p.parse_args()

# -------------------- Servidor de eventos TCP --------------------
class EventClients:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: Set[socket.socket] = set()

    def add(self, sock: socket.socket):
        with self._lock:
            self._clients.add(sock)

    def remove(self, sock: socket.socket):
        with self._lock:
            if sock in self._clients:
                self._clients.remove(sock)
            try:
                sock.close()
            except Exception:
                pass

    def broadcast(self, obj: dict):
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        dead = []
        with self._lock:
            for s in list(self._clients):
                try:
                    s.sendall(line)
                except Exception:
                    dead.append(s)
            for s in dead:
                try:
                    self._clients.remove(s)
                except KeyError:
                    pass
                try:
                    s.close()
                except Exception:
                    pass

event_clients = EventClients()

class EventHandler(socketserver.BaseRequestHandler):
    def handle(self):
        event_clients.add(self.request)
        try:
            hello = json.dumps({"type":"HELLO","info":"Eventos de rollos (DONE). Una línea JSON por evento."}, ensure_ascii=False) + "\n"
            self.request.sendall(hello.encode("utf-8"))
        except Exception:
            pass
        try:
            while True:
                data = self.request.recv(16)
                if not data:
                    break
        finally:
            event_clients.remove(self.request)

def start_event_server(host: str, port: int) -> Tuple[socketserver.ThreadingTCPServer, int]:
    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
    srv = ReusableTCPServer((host, port), EventHandler)
    real_port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    print(f"[SERVER] Eventos TCP escuchando en {host}:{real_port}")
    return srv, real_port

# -------------------- HTTP (dos streams MJPEG) --------------------
class AsyncFrameBus:
    def __init__(self, jpeg_quality: int = 80):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._last_jpeg: Optional[bytes] = None
        self._last_bgr: Optional[np.ndarray] = None
        self._q = []  # cola tamaño 1 (manual, sin queue para evitar locks extras)
        self._stop = False
        self._jpeg_quality = int(jpeg_quality)

        # TurboJPEG (opcional)
        self._tj = None
        try:
            from turbojpeg import TurboJPEG  # pip install turbojpeg
            self._tj = TurboJPEG()
        except Exception:
            self._tj = None

        self._th = threading.Thread(target=self._encoder_loop, daemon=True)
        self._th.start()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass

    def publish(self, bgr_frame: np.ndarray):
        # Copiamos un “snapshot” liviano para el display local y empujamos a la cola de encoder
        with self._cond:
            self._last_bgr = bgr_frame.copy()

            # Cola tamaño 1: si hay algo pendiente, lo pisamos (drop frame para bajar latencia)
            self._q = [bgr_frame.copy()]
            self._cond.notify_all()

    def _encode(self, bgr: np.ndarray) -> Optional[bytes]:
        if self._tj is not None:
            # TurboJPEG espera BGR->RGB normalmente; pero tiene flag para BGR
            try:
                return self._tj.encode(bgr, quality=self._jpeg_quality, pixel_format=1)  # 1 = TJPF_BGR
            except Exception:
                pass
        # Fallback OpenCV
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        return buf.tobytes() if ok else None

    def _encoder_loop(self):
        while True:
            with self._cond:
                while (not self._q) and (not self._stop):
                    self._cond.wait(timeout=0.5)
                if self._stop:
                    return
                bgr = self._q[-1]
                self._q.clear()
            # encode fuera del lock
            payload = self._encode(bgr)
            if payload is None:
                continue
            with self._cond:
                self._last_jpeg = payload
                self._cond.notify_all()

    def generator(self):
        boundary = b"--frame\r\n"
        while True:
            with self._cond:
                if self._last_jpeg is None:
                    self._cond.wait(timeout=1.0)
                payload = self._last_jpeg
            if payload is None:
                time.sleep(0.01)
                continue
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" +
                   payload + b"\r\n")

    def get_last_bgr(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        with self._cond:
            if self._last_bgr is None:
                self._cond.wait(timeout=timeout)
            return None if self._last_bgr is None else self._last_bgr.copy()

# instancias
frame_bus_cam1 = AsyncFrameBus(jpeg_quality=80)
frame_bus_cam2 = AsyncFrameBus(jpeg_quality=80)

def build_flask_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Roll Counter – Dual Stream</title></head>
<body style="background:#111;color:#eee;font-family:system-ui,Segoe UI,Roboto,Arial">
<h2>Roll Counter – Dual Stream</h2>
<p>Cam1 (derecha→izquierda): <code>/cam1</code> — Cam2 (izquierda→derecha): <code>/cam2</code></p>
<div style="display:flex;gap:12px;flex-wrap:wrap">
  <div><h3>Cam1</h3><img src="/cam1" style="max-width:48vw;border:1px solid #444"/></div>
  <div><h3>Cam2</h3><img src="/cam2" style="max-width:48vw;border:1px solid #444"/></div>
</div>
</body></html>
        """)

    @app.route("/cam1")
    def cam1():
        return Response(frame_bus_cam1.generator(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/cam2")
    def cam2():
        return Response(frame_bus_cam2.generator(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    return app

def start_http_server(app: Flask, host: str, port: int) -> Tuple[Any, int]:
    from werkzeug.serving import make_server
    srv = make_server(host, port, app, threaded=True)
    real_port = srv.socket.getsockname()[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    print(f"[SERVER] HTTP (Flask) escuchando en {host}:{real_port}")
    print(f"[SERVER] Abrí http://{host if host!='0.0.0.0' else 'localhost'}:{real_port}/ para ver ambos streams")
    return srv, real_port

# -------------------- Helpers tipografía (Pillow) --------------------
def _find_font_path(preferred: str | None) -> str | None:
    """Devuelve una ruta TTF válida o None. Intenta la preferida y algunas comunes."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "DejaVuSans.ttf", "Roboto-Regular.ttf", "NotoSans-Regular.ttf",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

def draw_text_pillow_inplace(img_bgr: np.ndarray, text: str, org: tuple[int,int],
                             font: ImageFont.FreeTypeFont | None,
                             color=(255,255,255), stroke=0, stroke_color=(0,0,0)):
    """Dibuja texto con Pillow sobre un array BGR SIN crear nueva imagen."""
    if font is None:
        x, y = org
        scale = 0.8
        thickness = 2 if stroke == 0 else max(2, stroke)
        cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    x, y = org
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    if stroke > 0:
        draw.text((x, y), text, font=font, fill=stroke_color,
                  stroke_width=stroke, stroke_fill=stroke_color)
    draw.text((x, y), text, font=font, fill=color)
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -------------------- Ventana local (mosaico superior + panel inferior con 2 columnas + botón REC) --------------------
class LocalDisplay(threading.Thread):
    def __init__(self, bus1: AsyncFrameBus, bus2: AsyncFrameBus, cam1, cam2, global_counter,
                 fps_cap: float = 0, slowdown: float = 1.0):
        super().__init__(daemon=True)
        self.bus1 = bus1
        self.bus2 = bus2
        self.cam1 = cam1
        self.cam2 = cam2
        self.global_counter = global_counter
        self.stop_evt = threading.Event()
        self.fps_cap = fps_cap
        self.slowdown = max(1.0, slowdown)

        # Tamaños de cada cámara
        self.tile_w = 640
        self.tile_h = 480
        self.gap = 4  # separación entre cámaras

        # Tipografía UI (Pillow). Usa --ui_font_path y --ui_font_size si están.
        font_path = _find_font_path(getattr(cam1.args, "ui_font_path", "") or "")
        if font_path is None:
            print("[WARN] No se encontró TTF. Fallback OpenCV para el panel. Pasá --ui_font_path con una ruta válida.")
        self._font_main = None
        self._font_small = None
        try:
            if font_path:
                ui_px = int(getattr(cam1.args, "ui_font_size", 18))
                self._font_main = ImageFont.truetype(font_path, max(10, ui_px))
                self._font_small = ImageFont.truetype(font_path, max(8, ui_px - 2))
        except Exception as e:
            print(f"[WARN] No pude cargar la fuente '{font_path}': {e}. Fallback OpenCV.")
            self._font_main = None
            self._font_small = None

        # Métricas del panel inferior (más alto, y 2 columnas)
        ui_px = int(getattr(cam1.args, "ui_font_size", 18))
        self.line_h = max(24, int(ui_px * 1.25))
        self.panel_h = max(240, int(ui_px * 7.2))

        self.order_str = "Pedido #A-12345"  # hardcodeado

        # --- Estado de grabación + botón ---
        self._recording = False
        self._btn_rect = (0, 0, 0, 0)  # (x1,y1,x2,y2)
        self._mouse_enabled = False

    def stop(self):
        self.stop_evt.set()

    @staticmethod
    def _put_label(img: np.ndarray, text: str):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (8, 10), (16 + tw, 10 + th + 12), (0, 0, 0), -1)
        cv2.putText(img, text, (14, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _fit_tile(self, frame: Optional[np.ndarray], label: str) -> np.ndarray:
        tile = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            msg = f"Esperando {label}…"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cx = max(10, (self.tile_w - tw) // 2)
            cy = max(30, (self.tile_h + th) // 2)
            cv2.putText(tile, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (self.tile_w-1, self.tile_h-1), (60, 60, 60), 1)
            return tile

        h, w = frame.shape[:2]
        scale = min(self.tile_w / w, self.tile_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (self.tile_w - new_w) // 2
        y0 = (self.tile_h - new_h) // 2
        tile[y0:y0+new_h, x0:x0+new_w] = resized
        cv2.rectangle(tile, (0, 0), (self.tile_w-1, self.tile_h-1), (60, 60, 60), 1)
        self._put_label(tile, label)
        return tile

    def _draw_button(self, canvas: np.ndarray):
        """Botón de Grabar/Detener + indicador REC."""
        H, W = canvas.shape[:2]
        pad = 12
        btn_w, btn_h = 170, 44
        x2 = W - pad
        x1 = x2 - btn_w
        y1 = pad
        y2 = y1 + btn_h
        self._btn_rect = (x1, y1, x2, y2)

        if self._recording:
            bg = (40, 40, 40)
            border = (0, 0, 255)
            text = "■ Detener"
            text_color = (240, 240, 240)
        else:
            bg = (60, 60, 60)
            border = (200, 200, 200)
            text = "● Grabar"
            text_color = (230, 230, 230)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        tx = x1 + (btn_w - tw) // 2
        ty = y1 + (btn_h + th) // 2 - 4
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.78, text_color, 2, cv2.LINE_AA)

        if self._recording:
            rec_text = "REC"
            (rw, rh), _ = cv2.getTextSize(rec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            rx = W - pad - rw - 26
            ry = self.tile_h - 12
            cv2.putText(canvas, rec_text, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.circle(canvas, (rx - 14, ry - rh // 2), 7, (0, 0, 255), -1)

    def _on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        x1, y1, x2, y2 = self._btn_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._toggle_recording()

    def _toggle_recording(self):
        """Activa/Desactiva la grabación en ambas cámaras."""
        self._recording = not self._recording
        if self._recording:
            self.cam1.set_rec_enabled(True)
            self.cam2.set_rec_enabled(True)
            print("[REC] Grabación iniciada (ambas cámaras).")
        else:
            self.cam1.set_rec_enabled(False)
            self.cam2.set_rec_enabled(False)
            # Asegurar cierre inmediato
            for cam in (self.cam1, self.cam2):
                try:
                    if cam.writer is not None:
                        cam.writer.release()
                        cam.writer = None
                        print(f"[REC] Video guardado en: {cam.out_path}")
                except Exception as e:
                    print(f"[WARN] Al cerrar writer de {cam.name}: {e}")
            print("[REC] Grabación detenida.")

    def _build_panel(self) -> np.ndarray:
        """
        Panel inferior:
          - Fila superior: encabezados (pedido, hora, contadores).
          - Abajo: dos columnas lado a lado con listas de rollos activos (izq=Cam1, der=Cam2).
        """
        W = self.tile_w * 2 + self.gap
        panel = np.full((self.panel_h, W, 3), 30, dtype=np.uint8)

        # Datos base
        hora = time.strftime("%H:%M:%S")
        total_global = self.global_counter["total"]
        now_m = time.monotonic()
        tmo = max(0.5, float(self.cam1.args.active_timeout))

        def is_active(t: 'TrackState') -> bool:
            return (not t.counted) and (t.last_seen_mono is not None) and ((now_m - t.last_seen_mono) <= tmo)

        activos_cam1 = len([t for t in list(self.cam1.tracks.values()) if is_active(t)])
        activos_cam2 = len([t for t in list(self.cam2.tracks.values()) if is_active(t)])
        activos_total = activos_cam1 + activos_cam2

        lh = self.line_h

        cv2.line(panel, (0, 0), (W, 0), (80, 80, 80), 1)

        y = 18
        draw_text_pillow_inplace(panel, f"🧾 {self.order_str}", (16, y), self._font_main,
                                 color=(255,255,255), stroke=2, stroke_color=(0,0,0)); y += lh
        draw_text_pillow_inplace(panel, f"⏰ {hora}", (16, y), self._font_main,
                                 color=(230,230,230), stroke=2, stroke_color=(0,0,0)); y += lh

        draw_text_pillow_inplace(panel, f"📦 Activos: {activos_total}", (16, y), self._font_main,
                                 color=(220,220,220), stroke=2, stroke_color=(0,0,0))
        draw_text_pillow_inplace(panel, f"✅ Pasados: {total_global}", (W//2 + 16, y), self._font_main,
                                 color=(200,255,200), stroke=2, stroke_color=(0,0,0))
        y += lh

        cv2.line(panel, (8, y), (W-8, y), (70, 70, 70), 1)
        y_lists_top = y + 10

        margin = 16
        inner_gap = 20
        col_w = (W - (margin*2) - inner_gap) // 2

        xL = margin
        xR = margin + col_w + inner_gap
        yL = y_lists_top
        yR = y_lists_top

        draw_text_pillow_inplace(panel, "Cam1 – Rollos activos", (xL, yL), self._font_main,
                                 color=(255,255,255), stroke=2, stroke_color=(0,0,0))
        draw_text_pillow_inplace(panel, "Cam2 – Rollos activos", (xR, yR), self._font_main,
                                 color=(255,255,255), stroke=2, stroke_color=(0,0,0))
        yL += lh
        yR += lh

        cv2.line(panel, (xL, yL), (xL + col_w, yL), (80,80,80), 1)
        cv2.line(panel, (xR, yR), (xR + col_w, yR), (80,80,80), 1)
        yL += 8
        yR += 8

        def dump_tracks_col(cam, x0: int, y0: int, y_max: int) -> int:
            y = y0
            for t in list(cam.tracks.values()):
                if not ((not t.counted) and (t.last_seen_mono is not None) and ((now_m - t.last_seen_mono) <= tmo)):
                    continue
                cx = f"{int(t.prev_cx):4d}" if t.prev_cx is not None else "----"
                cy = f"{int(t.prev_cy):4d}" if t.prev_cy is not None else "----"
                left_s  = fmt_ts(t.crossed_left_ts)
                right_s = fmt_ts(t.crossed_right_ts)

                draw_text_pillow_inplace(panel, f"id {t.id}  xy({cx},{cy})",
                                         (x0+6, y), self._font_small,
                                         color=(210,210,210), stroke=2, stroke_color=(0,0,0))
                y += int(lh * 0.9)
                draw_text_pillow_inplace(panel, f"L:{left_s}   R:{right_s}",
                                         (x0+6, y), self._font_small,
                                         color=(180,180,180), stroke=2, stroke_color=(0,0,0))
                y += int(lh * 0.9)

                if y > y_max - 14:
                    draw_text_pillow_inplace(panel, "…", (x0 + col_w - 24, y_max - 28), self._font_main,
                                             color=(200,200,200), stroke=2, stroke_color=(0,0,0))
                    break
            return y

        y_max = self.panel_h - 8
        dump_tracks_col(self.cam1, xL, yL, y_max)
        dump_tracks_col(self.cam2, xR, yR, y_max)

        cv2.line(panel, (xR - inner_gap//2, y_lists_top - 6), (xR - inner_gap//2, self.panel_h - 8), (60,60,60), 1)
        return panel

    def run(self):
        win = "Roll Counter – Dual + Panel 2 Columnas"
        total_w = self.tile_w*2 + self.gap
        total_h = self.tile_h + self.panel_h

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, total_w, total_h)

        # Fuerzo materializar la ventana antes del mouse callback (evita NULL handler)
        dummy = np.full((max(1,total_h), max(1,total_w), 3), 20, dtype=np.uint8)
        cv2.imshow(win, dummy)
        cv2.waitKey(1)

        # Intento registrar mouse
        self._mouse_enabled = False
        for _ in range(3):
            try:
                cv2.setMouseCallback(win, self._on_mouse)
                self._mouse_enabled = True
                break
            except cv2.error:
                cv2.imshow(win, dummy)
                cv2.waitKey(10)
                time.sleep(0.01)

        if self._mouse_enabled:
            print("[INFO] Botón en esquina superior derecha: '● Grabar' / '■ Detener'.")
        else:
            print("[WARN] Mouse callback no disponible en este backend; usar tecla 'r' para grabar/detener.")

        target_fps = self.fps_cap if (self.fps_cap and self.fps_cap > 0) else 30.0
        delay = max(1, int((1000.0 / max(1e-6, target_fps)) * self.slowdown))

        last1, last2 = None, None

        while not self.stop_evt.is_set():
            f1 = self.bus1.get_last_bgr(timeout=0.05)
            f2 = self.bus2.get_last_bgr(timeout=0.05)
            if f1 is not None: last1 = f1
            if f2 is not None: last2 = f2

            # Mosaico superior
            t1 = self._fit_tile(last1, "Cam1")
            t2 = self._fit_tile(last2, "Cam2")
            sep = np.full((self.tile_h, self.gap, 3), 30, dtype=np.uint8)
            mosaic = np.hstack([t1, sep, t2])

            # Panel inferior
            panel = self._build_panel()

            # Canvas final
            canvas = np.vstack([mosaic, panel])

            # Botón
            self._draw_button(canvas)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q')):
                self.stop_evt.set()
                break
            if key in (ord('r'), ord('R')):
                self._toggle_recording()

        try:
            cv2.destroyWindow(win)
        except Exception:
            pass

# -------------------- Fusión de eventos (deduplicación) --------------------
class FusionCoordinator(threading.Thread):
    """
    Recibe observaciones por cámara y emite un solo evento 'DONE' por rollo real.
    """
    def __init__(self, match_window: float, hold_window: float, csv_lock: threading.Lock, csv_f, global_counter: Dict[str, Any], epoch0_ms: int):
        super().__init__(daemon=True)
        self.match_window = float(match_window)
        self.hold_window = float(hold_window)
        self.csv_lock = csv_lock
        self.csv_f = csv_f
        self.global_counter = global_counter
        self.stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._pending: List[dict] = []
        self._next_roll_id = 1
        self.epoch0_ms = int(epoch0_ms)

        with self.csv_lock:
            self.csv_f.write("roll_id,cams,t_fused,t_cam1_start,t_cam1_end,t_cam2_start,t_cam2_end,dur_cam1_s,dur_cam2_s,count_global\n")
            self.csv_f.flush()

    def _to_epoch_ms(self, ts_sec: Optional[float]) -> Optional[int]:
        if ts_sec is None:
            return None
        return int(round(self.epoch0_ms + ts_sec * 1000.0))

    def stop(self):
        self.stop_evt.set()

    def _emit(self, cams: List[str], cam_obs: Dict[str, dict]):
        with self.global_counter["lock"]:
            roll_id = self._next_roll_id
            self._next_roll_id += 1
            self.global_counter["total"] += 1
            count_global = self.global_counter["total"]

        t1s = cam_obs.get("cam1", {}).get("start_ts")
        t1e = cam_obs.get("cam1", {}).get("end_ts")
        t2s = cam_obs.get("cam2", {}).get("start_ts")
        t2e = cam_obs.get("cam2", {}).get("end_ts")

        enters_sec = []
        exits_sec  = []
        if t1s is not None and t1e is not None:
            enters_sec.append(min(t1s, t1e))
            exits_sec.append(max(t1s, t1e))
        if t2s is not None and t2e is not None:
            enters_sec.append(min(t2s, t2e))
            exits_sec.append(max(t2s, t2e))

        enter_ms = self._to_epoch_ms(min(enters_sec)) if enters_sec else None
        exit_ms  = self._to_epoch_ms(max(exits_sec))  if exits_sec  else None

        evt = {
            "type": "DONE",
            "id": roll_id,
            "enter": enter_ms,
            "exit":  exit_ms,
            "count": 1,
            "total": count_global,
            "cams": cams
        }

        print(f"[DONE]  id={roll_id} ENTER={enter_ms if enter_ms is not None else 'NA'} "
              f"EXIT={exit_ms if exit_ms is not None else 'NA'} COUNT= 1 TOTAL={count_global}")

        event_clients.broadcast(evt)

        with self.csv_lock:
            d1 = (t1e - t1s) if (t1s is not None and t1e is not None) else ''
            d2 = (t2e - t2s) if (t2s is not None and t2e is not None) else ''
            mids = []
            if t1s is not None and t1e is not None: mids.append(0.5*(t1s+t1e))
            if t2s is not None and t2e is not None: mids.append(0.5*(t2s+t2e))
            t_fused = float(np.mean(mids)) if mids else None

            self.csv_f.write(
                f"{roll_id},{'&'.join(cams)},{fmt_ts(t_fused) if t_fused is not None else ''},"
                f"{fmt_ts(t1s) if t1s is not None else ''},{fmt_ts(t1e) if t1e is not None else ''},"
                f"{fmt_ts(t2s) if t2s is not None else ''},{fmt_ts(t2e) if t2e is not None else ''},"
                f"{d1},{d2},{count_global}\n"
            )
            self.csv_f.flush()

    def observe(self, camera: str, start_ts: float, end_ts: float, track_id: int):
        t_mid = 0.5*(start_ts + end_ts)
        obs = {
            "camera": camera,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "t_mid": t_mid,
            "track_id": track_id,
            "arrival": time.monotonic(),
        }
        other = "cam2" if camera == "cam1" else "cam1"

        with self._lock:
            match_i = None
            for i, p in enumerate(self._pending):
                if p["camera"] == other and abs(p["t_mid"] - t_mid) <= self.match_window:
                    match_i = i
                    break

            if match_i is not None:
                p = self._pending.pop(match_i)
                cam_obs = {camera: obs, other: p}
                cams = ["cam1","cam2"] if camera == "cam2" else ["cam1","cam2"]
                self._emit(cams, cam_obs)
            else:
                self._pending.append(obs)

    def run(self):
        while not self.stop_evt.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            to_emit: List[dict] = []
            with self._lock:
                still: List[dict] = []
                for p in self._pending:
                    if now - p["arrival"] >= self.hold_window:
                        to_emit.append(p)
                    else:
                        still.append(p)
                self._pending = still
            for p in to_emit:
                cam_obs = {p["camera"]: p}
                self._emit([p["camera"]], cam_obs)

# -------------------- Worker por cámara --------------------
class CameraWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        source: str,
        direction: str,              # "L2R" o "R2L"
        common_args,
        frame_bus: AsyncFrameBus,
        csv_lock: threading.Lock,
        csv_f,
        global_counter,
        fusion: FusionCoordinator,
        epoch0_ms: int
    ):
        super().__init__(daemon=True)
        self.name = name
        self.source = source
        self.direction = direction
        self.args = common_args
        self.frame_bus = frame_bus
        self.csv_lock = csv_lock
        self.csv_f = csv_f
        self.global_counter = global_counter
        self.local_total = 0
        self.should_stop = threading.Event()
        self.fusion = fusion
        self.epoch0_ms = int(epoch0_ms)

        # Grabación (controlado por UI)
        self.rec_enabled = False
        self.writer = None
        self.out_path = None

        model_path = latest_best(self.args.weights_glob, self.args.fallback_weights)
        self.model = YOLO(model_path)

        self.live_mode = is_stream_url(self.source)

        self.in_fps = 30.0
        self.W = None
        self.H = None
        if not self.live_mode:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[ERROR] ({self.name}) No se pudo abrir la fuente: {self.source}")
                raise SystemExit(1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.in_fps = float(fps)
            self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        if self.W and self.H:
            self._apply_geometry()
        else:
            self.left_x = self.right_x = self.band_top = self.band_bot = None

        self.tracks: Dict[int, TrackState] = {}

    # --- API para UI ---
    def set_rec_enabled(self, enabled: bool):
        self.rec_enabled = bool(enabled)
        if not self.rec_enabled:
            # cierre lazy: el writer se cierra en _maybe_close_writer() o por UI
            pass

    def _ensure_writer(self):
        if not self.rec_enabled:
            return
        if self.writer is not None:
            return
        # Crear carpeta y archivo
        out_dir = Path("runs_infer") / f"{self.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path_base = Path(self.args.out) if self.args.out else out_dir
        out_path_base.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path_base / f"{self.name}_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if self.W is None or self.H is None:
            # Si aún no conozco W/H, el primer frame las setea; salir hasta entonces
            return
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, max(1.0, self.in_fps), (int(self.W), int(self.H)))
        if self.writer.isOpened():
            print(f"[INFO][{self.name}] Grabando en: {self.out_path}")
        else:
            print(f"[WARN][{self.name}] No se pudo abrir VideoWriter para: {self.out_path}")
            self.writer = None

    def _maybe_close_writer(self):
        if self.rec_enabled:
            return
        if self.writer is not None:
            try:
                self.writer.release()
                print(f"[OK][{self.name}] Video guardado en: {self.out_path}")
            except Exception:
                pass
            self.writer = None

    def stop(self):
        self.should_stop.set()

    def _ts_ms_from_cam_seconds(self, ts_sec: float) -> int:
        return int(round(self.epoch0_ms + ts_sec * 1000.0))

    def _maybe_cross_events(self, st: TrackState, prev_cx: float, cx: float, frame_idx: int):
        if st.crossed_left_ts is None:
            if self.direction == "R2L" and (prev_cx is not None) and (self.left_x is not None) and (prev_cx > self.left_x >= cx):
                cross_ts = interp_cross_ts(prev_cx, cx, self.left_x, frame_idx - 1, max(1e-6, self.in_fps))
                st.crossed_left_ts = cross_ts
                self._emit_cross("LEFT", st.id, cross_ts)
            elif self.direction == "L2R" and (prev_cx is not None) and (self.left_x is not None) and (prev_cx < self.left_x <= cx):
                cross_ts = interp_cross_ts(prev_cx, cx, self.left_x, frame_idx - 1, max(1e-6, self.in_fps))
                st.crossed_left_ts = cross_ts
                self._emit_cross("LEFT", st.id, cross_ts)

        if st.crossed_right_ts is None:
            if self.direction == "R2L" and (prev_cx is not None) and (self.right_x is not None) and (prev_cx > self.right_x >= cx):
                cross_ts = interp_cross_ts(prev_cx, cx, self.right_x, frame_idx - 1, max(1e-6, self.in_fps))
                st.crossed_right_ts = cross_ts
                self._emit_cross("RIGHT", st.id, cross_ts)
            elif self.direction == "L2R" and (prev_cx is not None) and (self.right_x is not None) and (prev_cx < self.right_x <= cx):
                cross_ts = interp_cross_ts(prev_cx, cx, self.right_x, frame_idx - 1, max(1e-6, self.in_fps))
                st.crossed_right_ts = cross_ts
                self._emit_cross("RIGHT", st.id, cross_ts)

        if (st.crossed_left_ts is not None and st.crossed_right_ts is not None and not st.counted):
            st.counted = True
            self.local_total += 1
            start_ts = min(st.crossed_left_ts, st.crossed_right_ts)
            end_ts = max(st.crossed_left_ts, st.crossed_right_ts)
            self.fusion.observe(self.name, start_ts, end_ts, st.id)

    def _emit_cross(self, which: str, tid: int, cross_ts_sec: float):
        ts_ms = self._ts_ms_from_cam_seconds(cross_ts_sec)
        msg = {
            "type": "CROSS",
            "camera": self.name,
            "which": which,
            "track_id": tid,
            "ts_ms": ts_ms,
            "frame": int(round(cross_ts_sec * max(1e-6, self.in_fps)))
        }
        print(f"[CROSS][{self.name}] id={tid} LINE={which} ts_ms={ts_ms} frame~{msg['frame']}")
        event_clients.broadcast(msg)

    def run(self):
        out_dir = Path("runs_infer") / f"{self.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.t0_mono = None
        self.prev_frame_mono = None
        self.live_fps_ma = None

        frame_idx = 0
        while not self.should_stop.is_set():
            print(f"[INFO][{self.name}] Comenzando/rehaciendo inferencia…")
            ended_naturally = False

            if self.live_mode:
                if self.args.fps_cap and self.args.fps_cap > 0:
                    target_interval = 1.0 / max(0.1, float(self.args.fps_cap))
                else:
                    target_interval = 0.0
            else:
                src_fps = max(1e-6, self.in_fps)
                target_fps = src_fps / max(1.0, self.args.slowdown)
                if self.args.fps_cap and self.args.fps_cap > 0:
                    target_fps = min(target_fps, float(self.args.fps_cap))
                target_fps = max(0.1, target_fps)
                target_interval = 1.0 / target_fps
            next_due = time.perf_counter()

            for result in self.model.track(
                source=self.source,
                tracker=self.args.tracker,
                stream=True,
                conf=self.args.conf,
                imgsz=self.args.img,
                device=self.args.device,
                classes=self.args.classes,
                persist=True,
                verbose=False,
                show=False,
                stream_buffer=False,   # <<< evita buffering interno
                max_det=50,            # limita detecciones por frame (más rápido)
                iou=0.5,               # NMS razonable
                half=(self.args.device not in (None, "cpu")),  # FP16 en GPU
            ):

                if self.should_stop.is_set():
                    break

                frame = result.orig_img
                if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
                    if self.live_mode:
                        time.sleep(0.05)
                        continue
                    else:
                        ended_naturally = True
                        break

                if self.W is None or self.H is None:
                    self.H, self.W = frame.shape[:2]
                    self._apply_geometry()
                if self.live_mode and self.t0_mono is None:
                    self.t0_mono = time.monotonic()
                    self.prev_frame_mono = self.t0_mono

                if self.live_mode:
                    now_mono = time.monotonic()
                    ts_sec = now_mono - self.t0_mono
                    dt = max(1e-6, now_mono - self.prev_frame_mono)
                    inst_fps = 1.0 / dt
                    self.prev_frame_mono = now_mono
                    if self.live_fps_ma is None:
                        self.live_fps_ma = inst_fps
                    else:
                        self.live_fps_ma = 0.9 * self.live_fps_ma + 0.1 * inst_fps
                    if 2.0 <= self.live_fps_ma <= 120.0:
                        self.in_fps = float(self.live_fps_ma)
                else:
                    ts_sec = frame_idx / max(1e-6, self.in_fps)

                H, W = frame.shape[:2]

                # Writer: abrir/cerrar on-demand
                if self.rec_enabled and self.writer is None:
                    self._ensure_writer()
                if (not self.rec_enabled) and self.writer is not None:
                    self._maybe_close_writer()

                boxes = result.boxes
                if boxes is not None and boxes.id is not None and len(boxes) > 0:
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy().astype(float)

                    for det_i, tid in enumerate(ids):
                        x1, y1, x2, y2 = xyxy[det_i]
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)

                        st = self.tracks.get(tid)
                        if st is None:
                            st = TrackState(id=tid, first_seen_ts=ts_sec)
                            self.tracks[tid] = st

                        prev_cx = st.prev_cx
                        if prev_cx is not None:
                            self._maybe_cross_events(st, prev_cx, cx, frame_idx)

                        # actualizar última posición y última vez visto
                        st.prev_cx = cx
                        st.prev_cy = cy
                        st.last_seen_mono = time.monotonic()

                        color = (0, 255, 0) if st.counted else (0, 255, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{self.name}: id {tid}"
                        if (st.crossed_left_ts and not st.crossed_right_ts) or (st.crossed_right_ts and not st.crossed_left_ts):
                            label += " ▶"
                        elif st.counted:
                            label += " ✓"
                        cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # HUD + líneas
                if self.band_top is not None and self.band_bot is not None:
                    cv2.rectangle(frame, (0, self.band_top), (W, self.band_bot), (128, 128, 128), 1)
                if self.left_x is not None:
                    cv2.line(frame, (self.left_x, 0), (self.left_x, H), (0, 165, 255), 2)
                if self.right_x is not None:
                    cv2.line(frame, (self.right_x, 0), (self.right_x, H), (0, 165, 255), 2)
                hud_fps = self.in_fps if self.in_fps else 0.0
                hud = f"{self.name} count={self.local_total}  t={fmt_ts(ts_sec)}  FPS~{hud_fps:.1f}  dir={self.direction}"
                cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

                # rate limit
                if target_interval > 0:
                    now = time.perf_counter()
                    sleep_for = next_due - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                        now = next_due
                    next_due = max(now + target_interval, time.perf_counter())

                # >>> escribe SIEMPRE que esté grabando (no depende del target_interval)
                if self.rec_enabled and self.writer is not None:
                    self.writer.write(frame)

                # Stream HTTP + feed a display local
                self.frame_bus.publish(frame)

                # purga de tracks inactivos
                try:
                    now_m = time.monotonic()
                    tmo = max(0.5, float(self.args.active_timeout))
                    stale_ids = []
                    for tid, st in list(self.tracks.items()):
                        if st.counted:
                            continue
                        if st.last_seen_mono is None:
                            continue
                        if (now_m - st.last_seen_mono) > tmo:
                            stale_ids.append(tid)
                    for tid in stale_ids:
                        self.tracks.pop(tid, None)
                except Exception:
                    pass

                frame_idx += 1

            if self.should_stop.is_set():
                break

            if ended_naturally:
                print(f"[INFO][{self.name}] Fin del video/stream. Reiniciando desde el principio…")
                self.tracks.clear()
                frame_idx = 0
                self.t0_mono = None
                self.prev_frame_mono = None
                self.live_fps_ma = None
                continue

            print(f"[WARN][{self.name}] Loop terminado inesperadamente. Reintentando…")
            self.tracks.clear()
            frame_idx = 0
            self.t0_mono = None
            self.prev_frame_mono = None
            self.live_fps_ma = None
            continue

        # Cierre final de writer si quedó abierto
        self._maybe_close_writer()

    def _apply_geometry(self):
        # Elegir overrides según nombre de cámara
        if self.name == "cam1":
            left_ratio = self.args.left_ratio1 if self.args.left_ratio1 is not None else self.args.left_ratio
            right_ratio = self.args.right_ratio1 if self.args.right_ratio1 is not None else self.args.right_ratio
            band_top_ratio = self.args.band_top_ratio1 if self.args.band_top_ratio1 is not None else self.args.band_top_ratio
            band_bot_ratio = self.args.band_bot_ratio1 if self.args.band_bot_ratio1 is not None else self.args.band_bot_ratio
            band_slack_px = self.args.band_slack_px1 if self.args.band_slack_px1 is not None else self.args.band_slack_px
        else:  # cam2
            left_ratio = self.args.left_ratio2 if self.args.left_ratio2 is not None else self.args.left_ratio
            right_ratio = self.args.right_ratio2 if self.args.right_ratio2 is not None else self.args.right_ratio
            band_top_ratio = self.args.band_top_ratio2 if self.args.band_top_ratio2 is not None else self.args.band_top_ratio
            band_bot_ratio = self.args.band_bot_ratio2 if self.args.band_bot_ratio2 is not None else self.args.band_bot_ratio
            band_slack_px = self.args.band_slack_px2 if self.args.band_slack_px2 is not None else self.args.band_slack_px

        self.left_x = int(left_ratio * self.W)
        self.right_x = int(right_ratio * self.W)
        self.band_top = int(band_top_ratio * self.H) - band_slack_px
        self.band_bot = int(band_bot_ratio * self.H) + band_slack_px
        self.band_top = max(0, self.band_top)
        self.band_bot = min(self.H - 1, self.band_bot)

# -------------------- main --------------------
def main():
    args = parse_args()

    # Ctrl+C
    def handle_sigint(sig, frame):
        print("\n[INFO] Interrumpido por usuario.")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # Carpeta raíz de salida
    out_root = Path("runs_infer") / f"dual_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    # CSV unificado
    csv_path = out_root / "roll_events_fused.csv"
    csv_f = open(csv_path, "w", encoding="utf-8")

    # Servidor de eventos
    ev_srv, ev_port = start_event_server(args.host, args.event_port)

    # HTTP
    app = build_flask_app()
    http_srv, http_port = start_http_server(app, args.host, args.http_port)

    if args.show:
        print("[INFO] Modo dual con ventana local + panel (HTTP activo en paralelo).")

    # Estado global de conteo (único, post-fusión)
    global_counter = {"total": 0, "lock": threading.Lock()}
    csv_lock = threading.Lock()

    # Coordinador de fusión
    epoch0_ms = int(time.time() * 1000)
    fusion = FusionCoordinator(match_window=args.match_window,
                               hold_window=args.hold_window,
                               csv_lock=csv_lock, csv_f=csv_f,
                               global_counter=global_counter,
                               epoch0_ms=epoch0_ms)
    fusion.start()

    # Menos buffering en RTSP (FFmpeg via OpenCV)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|max_delay;500000|buffer_size;4096|stimeout;2000000"
    )
    # Si tus cámaras soportan TCP mejor que UDP, cambiá 'udp' por 'tcp'.

    # Workers de cámaras
    cam1 = CameraWorker(
        name="cam1",
        source=args.source1,
        direction=args.dir1,      # R2L por default
        common_args=args,
        frame_bus=frame_bus_cam1,
        csv_lock=csv_lock,
        csv_f=csv_f,
        global_counter=global_counter,
        fusion=fusion,
        epoch0_ms=epoch0_ms
    )
    cam2 = CameraWorker(
        name="cam2",
        source=args.source2,
        direction=args.dir2,      # L2R por default
        common_args=args,
        frame_bus=frame_bus_cam2,
        csv_lock=csv_lock,
        csv_f=csv_f,
        global_counter=global_counter,
        fusion=fusion,
        epoch0_ms=epoch0_ms
    )

    # Lanzar workers
    cam1.start()
    cam2.start()

    print(f"[READY] Eventos TCP en puerto {ev_port}. HTTP en puerto {http_port}.")
    print(f"[READY] Ver streams: /cam1 y /cam2 (portada en /). CSV (unificado): {csv_path}")

    if args.show:
        # IMPORTANTE: correr la UI en el hilo principal (Windows/HighGUI a veces lo exige)
        disp = LocalDisplay(frame_bus_cam1, frame_bus_cam2, cam1, cam2, global_counter,
                            fps_cap=args.fps_cap, slowdown=args.slowdown)
        try:
            disp.run()  # bloquea hasta cerrar la ventana (ESC o 'q')
        finally:
            # apagar todo ordenadamente
            cam1.stop(); cam2.stop()
            cam1.join(timeout=2.0); cam2.join(timeout=2.0)

            fusion.stop(); fusion.join(timeout=2.0)

            # Cerrar writers por si quedaron abiertos
            for cam in (cam1, cam2):
                try:
                    if cam.writer is not None:
                        cam.writer.release()
                        cam.writer = None
                except Exception:
                    pass

            # Cerrar CSV/servers
            csv_f.close()
            try: http_srv.shutdown()
            except Exception: pass
            try: http_srv.server_close()
            except Exception: pass
            try: ev_srv.shutdown()
            except Exception: pass
            try: ev_srv.server_close()
            except Exception: pass

            print(f"[OK] Total global (fused): {global_counter['total']}")
            print(f"[OK] CSV -> {csv_path}")
    else:
        # modo sin UI: solo mantener workers vivos
        try:
            while cam1.is_alive() or cam2.is_alive():
                time.sleep(0.5)
        finally:
            cam1.stop(); cam2.stop()
            cam1.join(timeout=2.0); cam2.join(timeout=2.0)

            fusion.stop(); fusion.join(timeout=2.0)

            for cam in (cam1, cam2):
                try:
                    if cam.writer is not None:
                        cam.writer.release()
                        cam.writer = None
                except Exception:
                    pass

            csv_f.close()
            try: http_srv.shutdown()
            except Exception: pass
            try: http_srv.server_close()
            except Exception: pass
            try: ev_srv.shutdown()
            except Exception: pass
            try: ev_srv.server_close()
            except Exception: pass

            print(f"[OK] Total global (fused): {global_counter['total']}")
            print(f"[OK] CSV -> {csv_path}")

if __name__ == "__main__":
    main()

'''
set KMP_DUPLICATE_LIB_OK=TRUE

python roll_dual_server.py ^
  --source1 "rtsp://admin:BKF456camara@192.168.36.213:554/cam/realmonitor?channel=1&subtype=0" --dir1 R2L ^
  --source2 "rtsp://admin:BKF456camara@192.168.36.214:554/cam/realmonitor?channel=1&subtype=0" --dir2 L2R ^
  --left_ratio1 0.35 --right_ratio1 0.70 --band_top_ratio1 0 --band_bot_ratio1 1 --band_slack_px1 20 ^
  --left_ratio2 0.4 --right_ratio2 0.70 --band_top_ratio2 0 --band_bot_ratio2 1 --band_slack_px2 20 ^
  --conf 0.30 --show

'''
