# Eye Gaze & Head Pose Intent Tracker
 
Real-time eye tracking, blink detection, and head-pose-based "intent"
estimation from a webcam or phone camera (e.g. DroidCam), with 9-point
screen calibration and detailed CSV logging.
 
Originally built on top of
[alireza787b/Python-Gaze-Face-Tracker](https://github.com/alireza787b/Python-Gaze-Face-Tracker),
extended with screen-space gaze calibration, robust head-pose math for
off-axis/extreme camera mounting, and CSV instrumentation aimed at
downstream intent classification (e.g. for assistive control of a
wheelchair or other device).
 
> **Status:** actively evolving. The current focus is getting gaze/head
> "intent" detection (forward/back/left/right) reliable when the camera is
> mounted off-angle — e.g. on a wheelchair, at a low or side angle, rather
> than a normal front-facing laptop webcam. Sending inferred intent to an
> external device (e.g. a wheelchair motor controller) is a planned next
> step and is **not** implemented yet in this codebase.
 
## Features
 
- Real-time face mesh (MediaPipe FaceMesh, 468 landmarks + iris refinement)
- Blink detection via Eye Aspect Ratio (EAR)
- Head pose estimation (pitch / yaw / roll) via `solvePnP` + a standard
  atan2-based Euler decomposition (monotonic and stable at extreme mounting
  angles — see [Notes on head pose math](#notes-on-head-pose-math))
- 9-point on-screen calibration that maps live eye/head features to actual
  screen coordinates, re-run automatically on every launch (and on demand
  with a hotkey) since a handheld/repositionable camera invalidates any
  previous calibration
- Discrete gaze/head "intent" direction (`Forward` / `Left` / `Right` /
  `Up` / `Down`) alongside continuous predicted screen X/Y
- Live on-screen overlay: blink count, pitch/yaw/roll, calibration
  quality (RMSE), predicted screen position, gaze direction
- Detailed per-frame CSV logging, only while actively recording
## Requirements
 
- **Python 3.9–3.11** (MediaPipe does not currently support 3.12+ reliably
  — this is the #1 cause of `AttributeError: module 'mediapipe' has no
  attribute 'solutions'` if you hit that)
- A webcam, or a phone acting as a webcam via
  [DroidCam](https://www.dev47apps.com/) or similar
See [`requirements.txt`](requirements.txt) for pinned package versions.
 
## Setup
 
```bash
# 1. Clone
git clone <this-repo-url>
cd <repo-folder>
 
# 2. Create a virtual environment with a supported Python version
py -3.11 -m venv venv          # Windows
# or: python3.11 -m venv venv  # macOS/Linux
 
# 3. Activate it
venv\Scripts\activate          # Windows
# or: source venv/bin/activate # macOS/Linux
 
# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
 
Verify MediaPipe is working before running the full script:
 
```bash
python -c "import mediapipe as mp; print(mp.solutions.face_mesh)"
```
 
## Usage
 
```bash
python detailed_eye_tracking_screen_calibrated.py
```
 
Optional camera source argument (index or DroidCam's virtual camera index):
 
```bash
python detailed_eye_tracking_screen_calibrated.py --camSource 1
```
 
### Calibration
 
On every launch, a 9-point calibration sequence runs first: dots appear on
your **computer screen** (not the camera), and you follow each one with
your eyes while keeping your head as still as possible. This is
deliberate and mandatory every run — calibration teaches the model the
mapping between "what the camera currently sees" and "where that
corresponds to on screen," which depends on wherever the camera happens to
be positioned *this session*. If the camera moves, the old mapping is
invalid.
 
- If calibration RMSE comes back high (default threshold: 150px, see
  `CALIBRATION_RMSE_WARNING_PIXELS`), you'll be prompted to redo it
  immediately.
- Press **`C`** at any time during tracking to redo the full calibration
  (e.g. if the camera/phone shifted mid-session).
### Hotkeys (during tracking)
 
| Key | Action |
|-----|--------|
| `C` | Redo full screen calibration |
| `R` | Start / stop CSV recording |
| `Q` | Quit |
 
### CSV Output
 
CSV files are written to `logs/eye_tracking_log_<timestamp>.csv` only for
the span of time between pressing `R` to start and `R` to stop. Each row
includes:
 
- Timing: timestamp, date/time, recording session number, elapsed time,
  frame number
- Per-eye: center position, width/height, pupil position, normalized
  horizontal/vertical pupil ratio
- Averaged pupil position/ratios across both eyes
- Eye Aspect Ratio (per eye + average), blink flag, running blink count
- Discrete gaze direction (horizontal / vertical / combined)
- Head pose: pitch, yaw, roll (relative to the calibrated neutral pose)
- Calibrated screen X/Y prediction, calibrated gaze direction, and the
  calibration RMSE that was in effect for that session
## Key configuration (top of the script)
 
| Constant | Purpose |
|---|---|
| `USER_FACE_WIDTH` | Your face width in mm — used to scale the 3D head model |
| `CALIBRATION_ENABLED` | Toggle the mandatory startup calibration |
| `CALIBRATION_RMSE_WARNING_PIXELS` | RMSE threshold that triggers a "redo calibration?" prompt |
| `HEAD_YAW_THRESHOLD` / `HEAD_PITCH_THRESHOLD` | Degrees of deviation from neutral before intent is classified as Left/Right/Up/Down instead of Forward |
| `INVERT_YAW` / `INVERT_PITCH` / `INVERT_ROLL` | Sign-flip toggles for head pose — **tune these empirically for your camera mounting** (see below) |
| `BLINK_THRESHOLD` / `EYE_AR_CONSEC_FRAMES` | Blink detection sensitivity |
| `LOG_FOLDER` | Where CSV logs are written |
 
### Tuning `INVERT_YAW` / `INVERT_PITCH` / `INVERT_ROLL`
 
The 3D face model's axis convention doesn't perfectly match OpenCV's
camera-frame convention, so the sign of pitch/yaw/roll coming out of the
solvePnP decomposition can come out mirrored depending on exactly how the
camera is mounted. There's no way to derive the correct sign purely from
code — test it:
 
1. Run the script, look **left**. If the "Face Looking at" text (or
   predicted Screen X) says "Right" instead, flip `INVERT_YAW`.
2. Look **down**. If it reports "Up" instead, flip `INVERT_PITCH`.
3. Tilt your head sideways (ear toward shoulder). If the printed roll
   value has the wrong sign, flip `INVERT_ROLL`.
4. **Recalibrate** (`C` or restart) after changing any of these — the
   neutral offset needs to be recaptured with the new sign convention.
## Notes on head pose math
 
Head pose (pitch/yaw/roll) is computed via `cv.solvePnP` followed by a
standard atan2-based rotation-matrix-to-Euler decomposition
(`rotation_matrix_to_euler_angles`). This replaced an earlier approach
using `cv.decomposeProjectionMatrix` plus a hand-tuned renormalization
step, which worked fine for a camera mounted level and centered on the
face, but could saturate or fold non-monotonically at extreme mounting
angles (e.g. a phone mounted low/off-axis on a wheelchair) — symptoms
looked like "always reports forward" or "direction reverses when tilting
further." The atan2-based decomposition is continuous and monotonic
across the full practical range and doesn't have this failure mode.
 
## Known limitations / roadmap
 
- Calibration assumes the head stays roughly still relative to the camera
  during the ~10–15 second calibration sequence; a camera that moves
  *during* calibration (not just between sessions) will still produce a
  bad mapping — redo calibration if RMSE is high.
- Intended for **research and accessibility prototyping**. It has not
  been validated for safety-critical control (e.g. driving a real
  wheelchair) — see [Safety notice](#safety-notice) below.
- Planned: streaming inferred intent to an external controller (e.g. over
  serial/Bluetooth/network) for downstream device control. Not yet
  implemented.
## Safety notice
 
This project estimates gaze/head-pose *intent* from a webcam feed for
research and prototyping purposes. It has **not** been validated,
certified, or tested for use as the sole control input for any
safety-critical system (including, but not limited to, powered
wheelchairs or other mobility devices). If you build on this for that
purpose:
 
- Always include an independent, physical emergency-stop mechanism that
  does **not** depend on this software.
- Never rely on gaze/head-pose inference alone to initiate movement
  without a deliberate, hard-to-trigger-by-accident confirmation step.
- Test extensively in a safe, supervised, unoccupied setting before any
  use with a person in the chair.
- This is not a substitute for a professionally engineered and certified
  assistive mobility control system.
## License
 
See [`LICENSE`](LICENSE).
 
## Acknowledgements
 
- [Alireza Bagheri](https://github.com/alireza787b) — original
  Python-Gaze-Face-Tracker project and blink/head-pose foundations.
- [Asadullah Dal](https://github.com/Asadullah-Dal17) — iris segmentation
  and blink detection techniques this project builds on.
