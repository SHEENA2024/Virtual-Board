import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
import time

# --- Hand Tracking Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# --- Drawing Settings ---
deque_len = 1024
drawing_enabled = False
current_color = (0, 0, 255)  # Red
current_brush_size = 3
color_points = defaultdict(lambda: deque(maxlen=deque_len))
cooldown_time = 1
last_button_time = time.time()

# --- Canvas Setup ---
persistent_canvas = np.ones((720, 1080, 3), dtype=np.uint8) * 255
cv2.namedWindow("Air Canvas", cv2.WINDOW_AUTOSIZE)
prev_x, prev_y = 0, 0
smoothing = 0.3

# --- Color Palette ---
color_palette = [
    ((0, 0, 255), "RED"),
    ((0, 255, 0), "GREEN"),
    ((255, 0, 0), "BLUE"),
    ((0, 255, 255), "YELLOW"),
    ((0, 0, 0), "BLACK"),
    ((255, 255, 255), "WHITE")
]

def draw_buttons(img):
    """Draws the top control bar with clear, palette, and save."""
    # Clear
    cv2.rectangle(img, (20, 10), (140, 60), (50, 50, 50), -1)
    cv2.putText(img, "CLEAR", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save
    cv2.rectangle(img, (500, 10), (620, 60), (100, 100, 100), -1)
    cv2.putText(img, "SAVE", (525, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Brush size indicator
    cv2.putText(img, f"Size: {current_brush_size}", (280, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # Draw color palette below buttons
    x_start = 20
    for i, (color, name) in enumerate(color_palette):
        x1 = x_start + i * 90
        x2 = x1 + 70
        cv2.rectangle(img, (x1, 80), (x2, 130), color, -1)
        border = (255, 255, 255) if color != current_color else (0, 0, 0)
        cv2.rectangle(img, (x1, 80), (x2, 130), border, 2)
        cv2.putText(img, name[0], (x1 + 25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, border, 2)

def save_canvas():
    filename = f"air_canvas_{int(time.time())}.png"
    cv2.imwrite(filename, persistent_canvas)
    print(f"✅ Canvas saved as {filename}")

def detect_color_selection(x, y):
    """Detect if finger points to a palette color."""
    global current_color
    if 80 <= y <= 130:
        for i, (color, name) in enumerate(color_palette):
            x1 = 20 + i * 90
            x2 = x1 + 70
            if x1 <= x <= x2:
                current_color = color
                print(f"🎨 Color changed to {name}")
                return True
    return False

def main():
    global current_color, drawing_enabled, prev_x, prev_y, color_points
    global last_button_time, current_brush_size, persistent_canvas

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        temp_canvas = persistent_canvas.copy()

        draw_buttons(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                smoothed_x = int(prev_x + smoothing * (x - prev_x))
                smoothed_y = int(prev_y + smoothing * (y - prev_y))
                prev_x, prev_y = smoothed_x, smoothed_y
                smoothed = (smoothed_x, smoothed_y)

                cv2.circle(frame, smoothed, 8, (0, 0, 0), -1)

                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                pinch_distance = np.hypot(thumb_x - x, thumb_y - y)
                drawing_enabled = pinch_distance >= 40

                # Check UI interactions
                if smoothed_y < 65:
                    now = time.time()
                    if now - last_button_time > cooldown_time:
                        if 20 <= smoothed_x <= 140:
                            color_points.clear()
                            persistent_canvas.fill(255)
                            print("🧹 Canvas cleared")
                        elif 500 <= smoothed_x <= 620:
                            save_canvas()
                        last_button_time = now
                elif detect_color_selection(smoothed_x, smoothed_y):
                    pass  # handled in function
                elif drawing_enabled:
                    color_points[current_color].appendleft(smoothed)
                else:
                    color_points[current_color].appendleft(None)
        else:
            color_points[current_color].appendleft(None)

        # Draw stored lines
        for color, points in color_points.items():
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv2.line(frame, points[i - 1], points[i], color, current_brush_size)
                cv2.line(temp_canvas, points[i - 1], points[i], color, current_brush_size)
                cv2.line(persistent_canvas, points[i - 1], points[i], color, current_brush_size)

        cv2.imshow("Air Canvas", temp_canvas)
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            current_brush_size = min(20, current_brush_size + 1)
        elif key == ord('-'):
            current_brush_size = max(1, current_brush_size - 1)
        elif key == ord('r'):
            current_color = (0, 0, 255)
        elif key == ord('g'):
            current_color = (0, 255, 0)
        elif key == ord('b'):
            current_color = (255, 0, 0)
        elif key == ord('y'):
            current_color = (0, 255, 255)
        elif key == ord('k'):
            current_color = (0, 0, 0)
        elif key == ord('w'):
            current_color = (255, 255, 255)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

