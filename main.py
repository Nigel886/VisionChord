import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pygame


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
    Demo 阶段仅识别三种手势：
    - Do：握拳（四指指尖 tip 的 y 坐标都低于 pip）
    - Mi：四指伸直 + 手掌整体“水平”（掌指根连线近似水平）
    - Sol：五指伸直 + 手掌近似“竖起”并“朝向摄像头”（掌面法线接近 z 轴）
    返回值：'Do' | 'Mi' | 'Sol' | 'None'
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

    # Mi：四指伸直 + 掌指根（index_mcp 到 pinky_mcp）连线近似水平
    index_mcp = lm3d[5]
    pinky_mcp = lm3d[17]
    mcp_vec = _vec(index_mcp, pinky_mcp)
    hand_width = _safe_norm(mcp_vec)
    mcp_y_diff = abs(float(index_mcp[1] - pinky_mcp[1]))
    hand_horizontal = (mcp_y_diff / hand_width) < 0.25

    if four_straight and hand_horizontal:
        return "Mi"

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
        "Em": (329.63, 392.00, 493.88),  # E G B
        "G": (196.00, 246.94, 392.00),   # G B D（用 B3, D4, G4 的组合近似）
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
    return pygame.sndarray.make_sound(audio)


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


def main() -> None:
    # 1) 初始化音频：建议先 pre_init 再 init 以获得更低延迟
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.init()
    try:
        pygame.mixer.init()
    except Exception as e:
        raise RuntimeError(f"pygame.mixer 初始化失败：{e}")

    # 2) 预留音频目录：把你的 C_chord.wav / Em_chord.wav / G_chord.wav 放这里即可
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")

    chord_sounds: Dict[str, pygame.mixer.Sound] = {
        "C": load_chord_sound("C", audio_dir),
        "Em": load_chord_sound("Em", audio_dir),
        "G": load_chord_sound("G", audio_dir),
    }

    gesture_to_chord = {
        "Do": "C",
        "Mi": "Em",
        "Sol": "G",
        "None": "",
    }

    # 3) 摄像头初始化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头：请检查是否被占用，或尝试更换摄像头编号。")

    # 4) MediaPipe Hands 初始化
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    debouncer = GestureDebouncer(stable_frames=5)
    last_triggered: str = "None"
    current_display_gesture: str = "None"
    current_display_chord: str = ""

    last_fps_t = time.time()
    fps = 0.0

    # 5) 主循环：采集 -> 识别 -> 渲染 -> 音频触发
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 镜像翻转：更符合“对镜子做手势”的直觉
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raw_gesture = "None"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 转为 numpy 3D（归一化坐标系）
            lm = np.array(
                [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                dtype=np.float32,
            )
            raw_gesture = classify_curwen_gesture(lm)

        stable_gesture, changed = debouncer.update(raw_gesture)
        chord = gesture_to_chord.get(stable_gesture, "")

        current_display_gesture = stable_gesture
        current_display_chord = chord

        # 音频触发规则：
        # - 从 None/其他 切到有效手势：立即播放
        # - 同一稳定手势保持：只触发一次（不重叠播放）
        if changed:
            if stable_gesture in ("Do", "Mi", "Sol") and stable_gesture != last_triggered:
                snd = chord_sounds.get(chord)
                if snd:
                    snd.stop()
                    snd.play()
                last_triggered = stable_gesture
            elif stable_gesture == "None":
                last_triggered = "None"

        # FPS（仅用于调试观感）
        now = time.time()
        dt = now - last_fps_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_fps_t = now

        # 6) UI：左上角显示手势与和弦
        overlay = f"Gesture: {current_display_gesture}   Chord: {current_display_chord}   FPS: {fps:.1f}"
        cv2.putText(
            frame,
            overlay,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("VisionChord - Curwen Hand Signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
