import cv2
import numpy as np
import time
import autopy
from cvzone.HandTrackingModule import HandDetector
from pythonosc import udp_client  # pip install python-osc

# ======================== SETUP ========================

UDP_IP = "127.0.0.1"
UDP_PORT = 7000

# FIX #8: Use proper OSC client instead of raw socket.
# TouchDesigner's OSC In CHOP expects OSC protocol, not plain text UDP.
# In TouchDesigner: add OSC In CHOP → set port to 7000.
# Gesture arrives at /gesture (int), zoom arrives at /zoom (float).
osc_client = udp_client.SimpleUDPClient(UDP_IP, UDP_PORT)

# Camera Setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)  # camera warmup

if not cap.isOpened():
    print("Camera 0 not available, trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Virtual Mouse Setup
wCam, hCam = 640, 480
frameR = 100         # Frame reduction border (active mouse area)
smoothening = 7      # Higher = smoother but more lag
click_delay = 0.5    # Seconds between clicks
last_click_time = 0

wScr, hScr = autopy.screen.size()
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# FIX #7: Preserve last gesture instead of resetting to 0 every frame.
# Initialise to -1 so the first real gesture always fires.
last_gesture = -1
pTime = 0

# Zoom
zoom_start_distance = None
zoom_level = 0.0

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)

print("=== Gesture Control Running ===")
print("LEFT hand  → switches TD prompt (index=G1, index+mid=G2, index+mid+ring=G3)")
print("RIGHT hand → virtual mouse (index+mid=move, index only=click)")
print("BOTH hands → zoom mode (spread/close to zoom in/out)")
print("Press Q to quit")
print("================================")

# ======================== MAIN LOOP ========================

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # ================= HAND DETECTION =================
    # FIX #1: findHands() ALWAYS returns (hands_list, img) tuple.
    # The original `if len(result) == 2` was checking tuple length (always 2),
    # NOT the number of hands — so single-hand detection was silently dropped.
    # FIX #2: flipType=True because we already flipped the frame above.
    # Without this, Left and Right hands are swapped.
    # FIX #3: draw=True so landmarks are visible for debugging.
    try:
        hands, frame = detector.findHands(frame, draw=True, flipType=True)
        # Sanitise: only keep valid hand dicts, max 2
        hands = [h for h in hands if isinstance(h, dict) and "lmList" in h and "type" in h]
        hands = hands[:2]
    except Exception as e:
        print(f"Detection error: {e}")
        hands = []

    # FIX #7: Don't reset gesture_value to 0 — carry forward last known gesture.
    # Only update it when a valid gesture is actually detected this frame.
    gesture_value = last_gesture

    # ======================== MODE DETECTION ========================

    # ── MODE 1: TWO HANDS = ZOOM ────────────────────────────────────
    if len(hands) == 2:
        try:
            center1 = np.array(hands[0].get("center", [320, 240]))
            center2 = np.array(hands[1].get("center", [320, 240]))
            distance = np.linalg.norm(center1 - center2)

            cv2.line(frame, tuple(map(int, center1)), tuple(map(int, center2)),
                     (0, 255, 255), 3)

            if zoom_start_distance is None:
                zoom_start_distance = distance
                print(f"ZOOM baseline set: {distance:.0f}px")

            # FIX #5: Bidirectional zoom — allow -1 to 1 so closing
            # hands (zoom out) is as valid as spreading (zoom in).
            distance_delta = distance - zoom_start_distance
            zoom_level = float(np.clip(distance_delta / 200.0, -1.0, 1.0))

            # FIX #8: Send via OSC — TD reads /zoom as a float channel.
            osc_client.send_message("/zoom", zoom_level)

            # UI
            direction = "IN" if zoom_level > 0 else ("OUT" if zoom_level < 0 else "—")
            cv2.putText(frame, f"ZOOM {direction}: {zoom_level:+.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Distance: {distance:.0f}px  |  Base: {zoom_start_distance:.0f}px",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(frame, "Spread = zoom IN   |   Close = zoom OUT",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        except Exception as e:
            print(f"Zoom error: {e}")

    # ── MODE 2: ONE HAND = GESTURE / MOUSE ──────────────────────────
    elif len(hands) == 1:
        zoom_start_distance = None  # reset zoom baseline when leaving 2-hand mode
        hand = hands[0]

        try:
            fingers = detector.fingersUp(hand)
            handType = hand["type"]
            lmList = hand["lmList"]

            # ── LEFT HAND: prompt switching gestures ──────────────
            if handType == "Left":
                # FIX #4: Remove thumb (index 0) from patterns.
                # Thumb detection in MediaPipe is unreliable — it fires
                # inconsistently depending on hand angle. Use fingers 1-4 only.
                #
                # Gesture 0 → index only      [0, 1, 0, 0, 0]
                # Gesture 1 → index + middle  [0, 1, 1, 0, 0]
                # Gesture 2 → index+mid+ring  [0, 1, 1, 1, 0]
                #
                # These map to OSC /gesture int 0, 1, 2 in TouchDesigner.

                if fingers == [0, 1, 0, 0, 0]:
                    gesture_value = 0
                    label = "GESTURE 1  (index)"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture_value = 1
                    label = "GESTURE 2  (index + middle)"
                elif fingers == [0, 1, 1, 1, 0]:
                    gesture_value = 2
                    label = "GESTURE 3  (index + middle + ring)"
                else:
                    label = "IDLE (no gesture matched)"

                cv2.putText(frame, f"LEFT: {label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {fingers}", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

            # ── RIGHT HAND: virtual mouse ─────────────────────────
            elif handType == "Right":
                cv2.putText(frame, "RIGHT: MOUSE MODE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

                # MOVE: index + middle up  [0, 1, 1, 0, 0]
                if fingers == [0, 1, 1, 0, 0]:
                    x1, y1 = int(lmList[8][0]), int(lmList[8][1])

                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                    # FIX #6: Clamp to screen bounds — autopy crashes if
                    # coordinates exceed screen dimensions.
                    x3 = float(np.clip(x3, 0, wScr - 1))
                    y3 = float(np.clip(y3, 0, hScr - 1))

                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    autopy.mouse.move(clocX, clocY)
                    cv2.circle(frame, (x1, y1), 10, (200, 100, 0), cv2.FILLED)
                    plocX, plocY = clocX, clocY

                    cv2.putText(frame, "MOVING", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

                # CLICK: index only  [0, 1, 0, 0, 0]
                elif fingers == [0, 1, 0, 0, 0]:
                    current_time = time.time()
                    if (current_time - last_click_time) > click_delay:
                        autopy.mouse.click()
                        last_click_time = current_time
                        print("CLICK")
                        x1, y1 = int(lmList[8][0]), int(lmList[8][1])
                        cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

                    cv2.putText(frame, "CLICKING", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                else:
                    cv2.putText(frame, "IDLE", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

        except Exception as e:
            print(f"Single hand error: {e}")

    # ── MODE 3: NO HANDS ─────────────────────────────────────────────
    else:
        zoom_start_distance = None
        cv2.putText(frame, "No hands detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # ── SEND GESTURE VIA OSC ─────────────────────────────────────────
    # Only fires when gesture actually changes, only in single-hand mode.
    # FIX #8: Sends proper OSC message — TD OSC In CHOP reads /gesture as int.
    if len(hands) == 1 and gesture_value != last_gesture and gesture_value != -1:
        osc_client.send_message("/gesture", int(gesture_value))
        print(f"OSC sent → /gesture {gesture_value}")
        last_gesture = gesture_value

    # ── FPS ──────────────────────────────────────────────────────────
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    cv2.putText(frame, "Q to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Active mode label (top right)
    mode_labels = {2: "MODE: ZOOM", 1: "MODE: MOUSE/GESTURE", 0: "MODE: IDLE"}
    mode_str = mode_labels.get(len(hands), "MODE: IDLE")
    (tw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, mode_str, (frame.shape[1] - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── CLEANUP ──────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Done.")