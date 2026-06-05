import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


@dataclass
class GestureResult:
    gesture_name: str
    chord_name: str


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-9 else 1e-9


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / _safe_norm(v1)
    v2n = v2 / _safe_norm(v2)
    c = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _finger_straight(lm3d: np.ndarray, mcp: int, pip: int, tip: int) -> bool:
    """
    使用“关节夹角”判断手指是否伸直：
    - 当 MCP->PIP 与 TIP->PIP 方向几乎相反（角度接近 180°）时，认为伸直
    """
    v1 = _vec(lm3d[mcp], lm3d[pip])
    v2 = _vec(lm3d[tip], lm3d[pip])
    return _angle_deg(v1, v2) > 160.0


def _finger_curled(lm3d: np.ndarray, mcp: int, pip: int, tip: int) -> bool:
    v1 = _vec(lm3d[mcp], lm3d[pip])
    v2 = _vec(lm3d[tip], lm3d[pip])
    return (_angle_deg(v1, v2) < 140.0) and (lm3d[tip][1] > lm3d[pip][1])


def _palm_normal(lm3d: np.ndarray) -> np.ndarray:
    """
    用 wrist, index_mcp, pinky_mcp 近似得到掌面法线方向（右手定则）。
    MediaPipe Hands 的 z 轴：数值越小（更负）通常表示越靠近摄像头。
    """
    wrist = lm3d[0]
    index_mcp = lm3d[5]
    pinky_mcp = lm3d[17]
    v1 = _vec(wrist, index_mcp)
    v2 = _vec(wrist, pinky_mcp)
    n = np.cross(v1, v2)
    return n / _safe_norm(n)


def classify_curwen_gesture(lm3d: np.ndarray) -> str:
    """
    返回值：'Do' | 'Re' | 'Mi' | 'Fa' | 'Sol' | 'La' | 'Ti' | 'None'
    """
    # 关键点索引参考：MediaPipe Hands 21 点
    # 0 wrist
    # 食指: 5(mcp), 6(pip), 8(tip)
    # 中指: 9(mcp), 10(pip), 12(tip)
    # 无名指: 13(mcp), 14(pip), 16(tip)
    # 小指: 17(mcp), 18(pip), 20(tip)
    # 拇指: 2(mcp), 3(ip), 4(tip)（拇指关节结构不同，这里只做简单判定）

    index_straight = _finger_straight(lm3d, 5, 6, 8)
    middle_straight = _finger_straight(lm3d, 9, 10, 12)
    ring_straight = _finger_straight(lm3d, 13, 14, 16)
    pinky_straight = _finger_straight(lm3d, 17, 18, 20)

    four_straight = index_straight and middle_straight and ring_straight and pinky_straight
    ring_curled = _finger_curled(lm3d, 13, 14, 16)
    pinky_curled = _finger_curled(lm3d, 17, 18, 20)
    index_curled = _finger_curled(lm3d, 5, 6, 8)
    middle_curled = _finger_curled(lm3d, 9, 10, 12)

    # Do：四指指尖 y 坐标都“低于”其 PIP（图像坐标系 y 向下增大）
    do_cond = (
        (lm3d[8][1] > lm3d[6][1])
        and (lm3d[12][1] > lm3d[10][1])
        and (lm3d[16][1] > lm3d[14][1])
        and (lm3d[20][1] > lm3d[18][1])
        and (not four_straight)
    )
    if do_cond:
        return "Do"

    re_cond = index_straight and middle_straight and ring_curled and pinky_curled
    if re_cond:
        return "Re"

    # Mi：四指伸直 + 掌指根（index_mcp 到 pinky_mcp）连线近似水平
    index_mcp = lm3d[5]
    pinky_mcp = lm3d[17]
    mcp_vec = _vec(index_mcp, pinky_mcp)
    hand_width = _safe_norm(mcp_vec)
    mcp_y_diff = abs(float(index_mcp[1] - pinky_mcp[1]))
    hand_horizontal = (mcp_y_diff / hand_width) < 0.25

    if four_straight and hand_horizontal:
        return "Mi"

    palm_center = (lm3d[0] + lm3d[5] + lm3d[17]) / 3.0
    thumb_tip = lm3d[4]
    thumb_mcp = lm3d[2]
    thumb_down = float(thumb_tip[1]) > float(thumb_mcp[1])
    thumb_far = _safe_norm(_vec(palm_center, thumb_tip)) > _safe_norm(_vec(palm_center, thumb_mcp)) * 1.6
    four_fist = (
        (lm3d[8][1] > lm3d[6][1])
        and (lm3d[12][1] > lm3d[10][1])
        and (lm3d[16][1] > lm3d[14][1])
        and (lm3d[20][1] > lm3d[18][1])
        and (not four_straight)
    )
    fa_cond = thumb_down and thumb_far and four_fist
    if fa_cond:
        return "Fa"

    # Sol：五指伸直 + 掌面法线接近摄像头方向（|z| 分量占主导）
    # 这里对拇指做宽松判定：tip(4) 相对 wrist(0) 的距离较大即认为伸出
    thumb_extended = _safe_norm(_vec(lm3d[0], lm3d[4])) > _safe_norm(_vec(lm3d[0], lm3d[2])) * 1.2
    five_straight = four_straight and thumb_extended

    n = _palm_normal(lm3d)
    palm_facing_camera = abs(float(n[2])) > (abs(float(n[0])) + abs(float(n[1]))) * 1.2

    # 同时要求手势“竖起”：wrist->middle_mcp 更偏向竖直方向（y 变化占主导）
    wrist = lm3d[0]
    middle_mcp = lm3d[9]
    v = _vec(wrist, middle_mcp)
    hand_upright = abs(float(v[1])) > abs(float(v[0])) * 1.2

    if five_straight and palm_facing_camera and hand_upright:
        return "Sol"

    tips = [4, 8, 12, 16, 20]
    all_tips_lower = all(float(lm3d[i][1]) > float(palm_center[1]) for i in tips)
    curved = (index_curled or middle_curled or ring_curled or pinky_curled) and (not four_straight)
    la_cond = all_tips_lower and curved
    if la_cond:
        return "La"

    ti_cond = (
        index_straight
        and (float(lm3d[8][1]) < float(lm3d[6][1]))
        and (float(lm3d[8][1]) < float(lm3d[5][1]))
        and (not middle_straight)
        and (not ring_straight)
        and (not pinky_straight)
        and (float(lm3d[12][1]) > float(lm3d[10][1]))
        and (float(lm3d[16][1]) > float(lm3d[14][1]))
        and (float(lm3d[20][1]) > float(lm3d[18][1]))
    )
    if ti_cond:
        return "Ti"

    return "None"


def generate_strum_chord(
    chord: str,
    sample_rate: int = 44100,
    duration_s: float = 0.6,
) -> pygame.mixer.Sound:
    """
    在没有音频文件时，动态合成一个“简单扫弦”音色：
    - 用多正弦叠加（和弦音）+ 轻微衰减包络
    - 目标是 Demo 可用，而不是专业音色
    """
    chord_freqs: Dict[str, Tuple[float, ...]] = {
        "C": (261.63, 329.63, 392.00),   # C E G
        "Dm": (293.66, 349.23, 440.00),  # D F A
        "Em": (329.63, 392.00, 493.88),  # E G B
        "F": (349.23, 440.00, 523.25),   # F A C
        "G": (196.00, 246.94, 392.00),   # G B D（用 B3, D4, G4 的组合近似）
        "Am": (220.00, 261.63, 329.63),  # A C E
        "G7": (196.00, 246.94, 293.66, 349.23),  # G B D F
    }
    freqs = chord_freqs.get(chord)
    if not freqs:
        freqs = (261.63, 329.63, 392.00)

    n = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)

    # 扫弦效果：给不同音符一个轻微的启动时间偏移
    offsets = [0.0, 0.02, 0.04, 0.06]
    sig = np.zeros_like(t, dtype=np.float32)
    for i, f in enumerate(freqs):
        off = offsets[i % len(offsets)]
        tt = np.clip(t - off, 0.0, duration_s)
        sig += np.sin(2.0 * math.pi * f * tt).astype(np.float32)
        sig += 0.2 * np.sin(2.0 * math.pi * (2.0 * f) * tt).astype(np.float32)

    # 衰减包络（快速起音 + 指数衰减）
    attack = int(sample_rate * 0.01)
    env = np.ones(n, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env *= np.exp(-3.5 * t).astype(np.float32)

    sig = sig * env
    sig /= max(float(np.max(np.abs(sig))), 1e-6)

    # pygame 需要 int16
    audio = (sig * 32767.0).astype(np.int16)
    mixer_init = pygame.mixer.get_init()
    channels = int(mixer_init[2]) if mixer_init else 2
    if channels == 1:
        return pygame.sndarray.make_sound(audio)

    if channels == 2:
        stereo = np.column_stack((audio, audio))
        return pygame.sndarray.make_sound(stereo)

    multi = np.repeat(audio[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(multi)


def load_chord_sound(chord: str, audio_dir: str) -> pygame.mixer.Sound:
    """
    优先加载音频文件；若不存在则生成简易合成音。
    预留文件名：
    - C -> C_chord.wav / C_chord.mp3
    - Em -> Em_chord.wav / Em_chord.mp3
    - G -> G_chord.wav / G_chord.mp3
    """
    candidates = []
    base = f"{chord}_chord"
    candidates.append(os.path.join(audio_dir, f"{base}.wav"))
    candidates.append(os.path.join(audio_dir, f"{base}.mp3"))

    for p in candidates:
        if os.path.exists(p):
            return pygame.mixer.Sound(p)

    return generate_strum_chord(chord)


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def draw_hand_landmarks(
    frame: np.ndarray,
    lm3d: np.ndarray,
    line_color: Tuple[int, int, int],
    point_color: Tuple[int, int, int],
) -> None:
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        ax, ay = int(lm3d[a][0] * w), int(lm3d[a][1] * h)
        bx, by = int(lm3d[b][0] * w), int(lm3d[b][1] * h)
        cv2.line(frame, (ax, ay), (bx, by), line_color, 2, cv2.LINE_AA)

    for i in range(lm3d.shape[0]):
        x, y = int(lm3d[i][0] * w), int(lm3d[i][1] * h)
        cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)


def ensure_hand_landmarker_model(model_path: str) -> None:
    """
    MediaPipe Tasks 版本需要手部模型文件（.task）。
    若本地不存在则自动下载到指定路径。
    """
    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        print(f"[VisionChord] Downloading hand_landmarker.task -> {model_path}")
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        raise RuntimeError(
            "无法自动下载 MediaPipe HandLandmarker 模型文件。\n"
            f"- 目标路径：{model_path}\n"
            f"- 下载地址：{url}\n"
            f"- 错误信息：{e}\n"
            "你也可以手动下载后放到该路径，然后重新运行。"
        )


class GestureDebouncer:
    """
    状态防抖（State Debounce）：
    - 只有当手势连续保持 N 帧以上，才认为切换成功
    - 同一手势保持时只触发一次（单次触发模式）
    """

    def __init__(self, stable_frames: int = 5):
        self.stable_frames = stable_frames
        self._candidate: str = "None"
        self._count: int = 0
        self._stable: str = "None"

    def update(self, new_gesture: str) -> Tuple[str, bool]:
        """
        返回 (stable_gesture, changed)
        changed=True 表示稳定态发生了变化
        """
        if new_gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = new_gesture
            self._count = 1

        if self._count >= self.stable_frames and self._stable != self._candidate:
            self._stable = self._candidate
            return self._stable, True

        return self._stable, False


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = float(alpha)
        self._value: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._value is None:
            self._value = value.copy()
            return self._value
        self._value = (self.alpha * value) + ((1.0 - self.alpha) * self._value)
        return self._value


class StrumDetector:
    def __init__(self):
        self._last_wrist_y: Optional[float] = None
        self._last_strum_t: float = 0.0
        self._pinch_closed: bool = False
        self._last_pinch_t: float = 0.0

    def update(self, lm3d: np.ndarray, now_s: float) -> Tuple[bool, bool]:
        wrist_y = float(lm3d[0][1])
        strum = False
        pinch = False

        if self._last_wrist_y is not None:
            dy = wrist_y - self._last_wrist_y
            if dy > 0.10 and (now_s - self._last_strum_t) > 0.25:
                strum = True
                self._last_strum_t = now_s
        self._last_wrist_y = wrist_y

        thumb_tip = lm3d[4]
        index_tip = lm3d[8]
        hand_scale = _safe_norm(_vec(lm3d[0], lm3d[9]))
        pinch_ratio = _safe_norm(_vec(thumb_tip, index_tip)) / max(hand_scale, 1e-6)
        is_closed = pinch_ratio < 0.35
        is_open = pinch_ratio > 0.45

        if is_closed:
            self._pinch_closed = True
        elif is_open and self._pinch_closed and (now_s - self._last_pinch_t) > 0.25:
            pinch = True
            self._last_pinch_t = now_s
            self._pinch_closed = False

        return strum, pinch


def _hand_label(results, i: int) -> str:
    try:
        handedness = results.handedness[i]
        if handedness and handedness[0]:
            c = handedness[0]
            label = getattr(c, "category_name", None) or getattr(c, "display_name", None) or ""
            return str(label)
    except Exception:
        return ""
    return ""


def main() -> None:
    # 1) 初始化音频：建议先 pre_init 再 init 以获得更低延迟
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.init()
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except Exception as e:
        raise RuntimeError(f"pygame.mixer 初始化失败：{e}")

    # 2) 预留音频目录：把你的 C_chord.wav / Em_chord.wav / G_chord.wav 放这里即可
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")

    chord_sounds: Dict[str, pygame.mixer.Sound] = {
        "C": load_chord_sound("C", audio_dir),
        "Dm": load_chord_sound("Dm", audio_dir),
        "Em": load_chord_sound("Em", audio_dir),
        "F": load_chord_sound("F", audio_dir),
        "G": load_chord_sound("G", audio_dir),
        "Am": load_chord_sound("Am", audio_dir),
        "G7": load_chord_sound("G7", audio_dir),
    }

    gesture_to_chord = {
        "Do": "C",
        "Re": "Dm",
        "Mi": "Em",
        "Fa": "F",
        "Sol": "G",
        "La": "Am",
        "Ti": "G7",
        "None": "",
    }

    # 3) 摄像头初始化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头：请检查是否被占用，或尝试更换摄像头编号。")

    # 4) MediaPipe 手部关键点检测初始化（Tasks API）
    # 说明：你安装的 mediapipe 版本不再提供 mp.solutions.*（旧接口），因此这里用 HandLandmarker（新接口）。
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
    ensure_hand_landmarker_model(model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)

    debouncer = GestureDebouncer(stable_frames=5)
    current_display_gesture: str = "None"
    selected_chord: str = ""
    chord_change_t: float = 0.0
    right_trigger_t: float = 0.0

    left_smoother = LandmarkSmoother(alpha=0.65)
    right_smoother = LandmarkSmoother(alpha=0.65)
    right_strum = StrumDetector()

    last_fps_t = time.time()
    fps = 0.0

    window_name = "VisionChord - Curwen Hand Signs"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            now_s = time.time()

            left_lm: Optional[np.ndarray] = None
            right_lm: Optional[np.ndarray] = None

            if results.hand_landmarks:
                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
                    label = _hand_label(results, i).lower()
                    if "left" in label:
                        left_lm = left_smoother.update(lm)
                    elif "right" in label:
                        right_lm = right_smoother.update(lm)
                    elif left_lm is None:
                        left_lm = left_smoother.update(lm)
                    elif right_lm is None:
                        right_lm = right_smoother.update(lm)

            if left_lm is not None:
                draw_hand_landmarks(frame, left_lm, (255, 180, 0), (255, 255, 255))
                raw_gesture = classify_curwen_gesture(left_lm)
            else:
                raw_gesture = "None"

            stable_gesture, changed = debouncer.update(raw_gesture)
            current_display_gesture = stable_gesture

            if changed:
                new_chord = gesture_to_chord.get(stable_gesture, "")
                if new_chord != selected_chord:
                    selected_chord = new_chord
                    chord_change_t = now_s

            if right_lm is not None:
                draw_hand_landmarks(frame, right_lm, (255, 80, 255), (255, 255, 255))
                did_strum, did_pinch = right_strum.update(right_lm, now_s)
                if (did_strum or did_pinch) and selected_chord:
                    snd = chord_sounds.get(selected_chord)
                    if snd:
                        snd.stop()
                        snd.play()
                    right_trigger_t = now_s
            elif changed and stable_gesture in ("Do", "Re", "Mi", "Fa", "Sol", "La", "Ti") and selected_chord:
                snd = chord_sounds.get(selected_chord)
                if snd:
                    snd.stop()
                    snd.play()
                right_trigger_t = now_s

            now = time.time()
            dt = now - last_fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_fps_t = now

            overlay = f"Gesture: {current_display_gesture}   Chord: {selected_chord or '-'}   FPS: {fps:.1f}"
            cv2.putText(
                frame,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )

            if selected_chord and (now_s - chord_change_t) < 0.35:
                hud = frame.copy()
                cv2.rectangle(hud, (0, 0), (frame.shape[1], 70), (255, 120, 0), -1)
                frame[:] = cv2.addWeighted(hud, 0.25, frame, 0.75, 0)
                cv2.putText(
                    frame,
                    f"{selected_chord}",
                    (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (255, 255, 255),
                    4,
                    cv2.LINE_AA,
                )

            if (now_s - right_trigger_t) < 0.20:
                cv2.putText(
                    frame,
                    "TRIGGER",
                    (frame.shape[1] - 210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
