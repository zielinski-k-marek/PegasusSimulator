"""
StreamingBackend: Streams camera frames via FFmpeg NVENC over UDP + records H.265.

Multi-camera support:
  - One camera (stream_camera) streams H.264 over UDP + records H.265
  - All other cameras record H.265 only (separate FFmpeg processes)
  - Camera identified by data["camera_name"] from MonocularCamera

The receiver reads the live stream with cv2.VideoCapture('udp://0.0.0.0:<port>').
"""

import os
import shutil
import subprocess
import time
import numpy as np

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


class StreamingBackendConfig(BackendConfig):

    def __init__(self, target_ip: str = "127.0.0.1", port: int = 5600,
                 bitrate: str = "4M", use_nvenc: bool = True,
                 record_dir: str = None, stream_camera: str = "camera"):
        super().__init__()
        self.target_ip = target_ip
        self.port = port
        self.bitrate = bitrate
        self.use_nvenc = use_nvenc
        self.record_dir = record_dir  # Directory for H.265 recordings (None = no recording)
        self.stream_camera = stream_camera  # Camera name to stream via UDP


class StreamingBackend(Backend):

    def __init__(self, config: StreamingBackendConfig = None):
        if config is None:
            config = StreamingBackendConfig()
        super().__init__(config)

        # Main stream camera (UDP + record)
        self._ffmpeg_proc = None
        self._frame_count = 0
        self._resolution = None
        self._started = False
        self._record_path = None

        # Record-only cameras: {camera_name: {"proc", "frame_count", "record_path", "started"}}
        self._recorders = {}

    def _find_ffmpeg(self) -> str:
        path = shutil.which("ffmpeg")
        if path:
            return path
        # Fallback: common Windows install locations
        for candidate in [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
        ]:
            if os.path.isfile(candidate):
                return candidate
        return "ffmpeg"  # hope it's on PATH

    def _make_record_path(self, camera_name: str = "flight") -> str:
        """Generate timestamped recording path in record_dir."""
        rec_dir = self.config.record_dir
        os.makedirs(rec_dir, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(rec_dir, f"{camera_name}_{ts}.mp4")

    def _start_ffmpeg(self, width: int, height: int, fps: int):
        """Start the main streaming FFmpeg (UDP stream + optional record)."""
        ffmpeg = self._find_ffmpeg()
        target = f"udp://{self.config.target_ip}:{self.config.port}?pkt_size=1316"

        # Input args (common)
        input_args = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
        ]

        # Output 1: H.264 low-latency UDP stream
        if self.config.use_nvenc:
            stream_args = [
                "-map", "0:v",
                "-c:v", "h264_nvenc",
                "-preset", "p1",       # fastest NVENC preset
                "-tune", "ull",        # ultra-low-latency
                "-rc", "cbr",          # constant bitrate for smooth stream
                "-b:v", self.config.bitrate,
                "-bf", "0",            # no B-frames (lower latency)
                "-g", str(fps),        # keyframe every second
                "-f", "mpegts",
                target,
            ]
        else:
            stream_args = [
                "-map", "0:v",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-b:v", self.config.bitrate,
                "-bf", "0",
                "-g", str(fps),
                "-f", "mpegts",
                target,
            ]

        # Output 2: H.265 recording to disk (optional)
        record_args = []
        if self.config.record_dir:
            self._record_path = self._make_record_path("flight")
            record_args = self._make_record_args(self._record_path)

        cmd = [*input_args, *stream_args, *record_args]

        print(f"[StreamingBackend] Starting FFmpeg: {' '.join(cmd)}")
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._started = True
        print(f"[StreamingBackend] Streaming to {target} ({width}x{height} @ {fps}Hz)")
        if self._record_path:
            print(f"[StreamingBackend] Recording H.265 to {self._record_path}")

    def _make_record_args(self, record_path: str) -> list:
        """Build FFmpeg output args for H.265 recording."""
        if self.config.use_nvenc:
            return [
                "-map", "0:v",
                "-c:v", "hevc_nvenc",
                "-preset", "p4",       # quality preset for recording
                "-rc", "vbr",
                "-cq", "28",           # constant quality
                "-bf", "0",
                "-pix_fmt", "yuv420p", # force standard pixel format (avoid gbrp)
                "-tag:v", "hvc1",      # Apple/browser compatible tag
                record_path,
            ]
        else:
            return [
                "-map", "0:v",
                "-c:v", "libx265",
                "-preset", "fast",
                "-crf", "28",
                "-pix_fmt", "yuv420p", # force standard pixel format
                "-tag:v", "hvc1",
                record_path,
            ]

    def _start_recorder(self, camera_name: str, width: int, height: int, fps: int):
        """Start a record-only FFmpeg process for a non-streamed camera."""
        if not self.config.record_dir:
            return

        ffmpeg = self._find_ffmpeg()
        record_path = self._make_record_path(camera_name)

        input_args = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
        ]
        record_args = self._make_record_args(record_path)
        cmd = [*input_args, *record_args]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._recorders[camera_name] = {
            "proc": proc,
            "frame_count": 0,
            "record_path": record_path,
            "started": True,
        }
        print(f"[StreamingBackend] Recording {camera_name}: {record_path} ({width}x{height} @ {fps}Hz)")

    def start(self):
        pass

    def stop(self):
        # Stop main stream
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=5)
            except Exception:
                self._ffmpeg_proc.kill()
            print(f"[StreamingBackend] Stopped stream. {self._frame_count} frames.")
            if self._record_path and os.path.isfile(self._record_path):
                size_mb = os.path.getsize(self._record_path) / (1024 * 1024)
                print(f"[StreamingBackend] Recording saved: {self._record_path} ({size_mb:.1f} MB)")
            self._ffmpeg_proc = None

        # Stop all record-only cameras
        for cam_name, rec in self._recorders.items():
            proc = rec["proc"]
            if proc is not None:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                print(f"[StreamingBackend] {cam_name}: {rec['frame_count']} frames recorded.")
                rp = rec["record_path"]
                if rp and os.path.isfile(rp):
                    size_mb = os.path.getsize(rp) / (1024 * 1024)
                    print(f"[StreamingBackend] Recording saved: {rp} ({size_mb:.1f} MB)")
        self._recorders = {}

    def reset(self):
        self.stop()
        self._frame_count = 0
        self._started = False

    def update_sensor(self, sensor_type: str, data):
        pass  # Sensors handled by ROS2Backend

    def update_graphical_sensor(self, sensor_type: str, data):
        if sensor_type != "MonocularCamera":
            return
        if data is None or "image" not in data:
            return

        image = data["image"]  # numpy RGB uint8, shape (H, W, 3)
        if image is None:
            return

        camera_name = data.get("camera_name", "camera")

        # Route: is this the stream camera or a record-only camera?
        if camera_name == self.config.stream_camera:
            self._handle_stream_camera(image, data)
        else:
            self._handle_record_camera(camera_name, image, data)

    def _handle_stream_camera(self, image, data):
        """Handle the main streaming camera (UDP + record)."""
        # Start ffmpeg on first frame (we need actual resolution)
        if not self._started:
            h, w = image.shape[:2]
            fps = data.get("frequency", 15)
            self._resolution = (w, h)
            self._start_ffmpeg(w, h, fps)

        # Diagnostic: check if source frames are black (first 5 frames)
        if self._frame_count < 5:
            mean_val = float(np.mean(image))
            max_val = float(np.max(image))
            print(f"[StreamingBackend] Frame #{self._frame_count}: "
                  f"shape={image.shape} dtype={image.dtype} "
                  f"mean={mean_val:.1f} max={max_val:.0f} "
                  f"{'BLACK' if max_val == 0 else 'OK'}", flush=True)

        # Pipe raw frame to ffmpeg stdin
        try:
            self._ffmpeg_proc.stdin.write(image.tobytes())
            self._frame_count += 1
        except (BrokenPipeError, OSError):
            # ffmpeg died — read stderr for diagnostics
            if self._ffmpeg_proc:
                err = self._ffmpeg_proc.stderr.read().decode(errors="replace")
                print(f"[StreamingBackend] FFmpeg died after {self._frame_count} frames")
                if err:
                    print(f"[StreamingBackend] FFmpeg stderr: {err[-500:]}")
                self._ffmpeg_proc = None
                self._started = False

    def _handle_record_camera(self, camera_name, image, data):
        """Handle a record-only camera."""
        if not self.config.record_dir:
            return

        # Start recorder on first frame
        if camera_name not in self._recorders:
            h, w = image.shape[:2]
            fps = data.get("frequency", 30)
            self._start_recorder(camera_name, w, h, fps)

        rec = self._recorders.get(camera_name)
        if rec is None or rec["proc"] is None:
            return

        try:
            rec["proc"].stdin.write(image.tobytes())
            rec["frame_count"] += 1
        except (BrokenPipeError, OSError):
            proc = rec["proc"]
            if proc:
                err = proc.stderr.read().decode(errors="replace")
                print(f"[StreamingBackend] {camera_name} FFmpeg died after {rec['frame_count']} frames")
                if err:
                    print(f"[StreamingBackend] {camera_name} FFmpeg stderr: {err[-500:]}")
                rec["proc"] = None

    def update_state(self, state):
        pass  # State handled by ROS2Backend

    def input_reference(self):
        return []  # Control handled by other backend

    def update(self, dt: float):
        pass
