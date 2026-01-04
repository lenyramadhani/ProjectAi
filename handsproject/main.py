import cv2
import mediapipe as mp

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture = ""

def y(landmarks, id): return landmarks[id].y
def x(landmarks, id): return landmarks[id].x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = ""

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),  # Titik hitam
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)               # Garis kuning
            )

            lm = hand_landmarks.landmark

            # === Halo ===
            all_fingers_up = all(
                y(lm, tip) < y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_fingers_up:
                gesture = "Halo"

            # === OK ===
            thumb_index_close = (
                abs(x(lm, mp_hands.HandLandmark.THUMB_TIP) - x(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05 and
                abs(y(lm, mp_hands.HandLandmark.THUMB_TIP) - y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05
            )
            middle_up = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            if thumb_index_close and middle_up:
                gesture = "OK"

            # === I Love You ===
            love_you = (
                y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP) and
                y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.PINKY_TIP) < y(lm, mp_hands.HandLandmark.PINKY_PIP) and
                y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) > y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            )
            if love_you:
                gesture = "I Love You"

            # === Peace (2 jari ke atas, lainnya turun) - support 2 arah
            index_up = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_up = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            ring_down = y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            pinky_down = y(lm, mp_hands.HandLandmark.PINKY_TIP) > y(lm, mp_hands.HandLandmark.PINKY_PIP)
            thumb_down = y(lm, mp_hands.HandLandmark.THUMB_TIP) > y(lm, mp_hands.HandLandmark.THUMB_IP)

            if index_up and middle_up and ring_down and pinky_down and thumb_down:
                gesture = "Peace"

            # === Fist ===
            all_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_folded:
                gesture = "Fist"

            # === Sip (Jempol) - support 2 sisi ===
            thumb_up = y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP)
            others_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if thumb_up and others_folded and not all_folded:
                gesture = "Sip"

    # Tampilkan teks di layar
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 3)  # Warna teks hitam

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()