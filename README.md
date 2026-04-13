Hand Gesture-Controlled Virtual Mouse & TouchDesigner Interface
This project is a real-time, touchless human-computer interaction system that replaces traditional input devices — mouse and keyboard — with natural hand gestures captured through a standard webcam. It sits at the intersection of computer vision, AI-based hand tracking, and creative software integration.

How It Works
At its core, the system uses OpenCV for live video capture and frame processing, combined with cvzone's HandTrackingModule (built on Google's MediaPipe) to detect and track up to two hands simultaneously. Every frame, it identifies which fingers are raised and interprets specific finger combinations as commands — all in real time.

The system operates in three intelligent modes based on how many hands are visible:

Right Hand (Mouse Mode): Raising the index and middle fingers together moves the mouse cursor across the screen, with smoothing applied to eliminate jitter. Raising only the index finger triggers a left click. Finger tip coordinates are mapped from the camera frame to full screen dimensions using interpolation, and coordinates are clamped to prevent crashes at screen edges.

Left Hand (Gesture Mode): Different finger combinations — one, two, or three fingers — correspond to gestures that are broadcast as OSC messages to TouchDesigner, a node-based visual programming environment. This allows the user to switch between visual scenes, prompts, or interactive states without touching anything.

Both Hands (Zoom Mode): When both hands are detected, the system measures the distance between them. Spreading hands apart signals zoom-in; bringing them together signals zoom-out. This delta is normalized and sent continuously to TouchDesigner as a float value over OSC.

Why It Matters
The project demonstrates how AI-powered gesture recognition can serve as a practical bridge between the physical and digital worlds — enabling accessible, immersive, and hands-free control for presentations, interactive art installations, or accessibility-focused applications.
