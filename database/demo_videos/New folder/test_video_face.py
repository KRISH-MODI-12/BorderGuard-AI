from pathlib import Path
import cv2
import re
import math

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

FACE_DIR = BASE / "database" / "authorized_faces"
VIDEO_DIR = BASE / "database" / "demo_videos"
OUTPUT_DIR = VIDEO_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test the first problematic video
VIDEO_PATH = VIDEO_DIR / "20260918_112954.mp4"

YUNET_PATH = BASE / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = BASE / "face_recognition_sface_2021dec.onnx"

FACE_THRESHOLD = 0.45

# ============================================================
# LOAD MODELS
# ============================================================

print()
print("==============================================")
print(" VIDEO FACE RECOGNITION DIAGNOSTIC")
print("==============================================")
print()

print("[1] Loading YuNet...")
detector = cv2.FaceDetectorYN.create(
    str(YUNET_PATH),
    "",
    (320, 320),
    0.70,
    0.30,
    5000
)

print("[OK] YuNet loaded")

print()
print("[2] Loading SFace...")
recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_PATH),
    ""
)

print("[OK] SFace loaded")

# ============================================================
# IDENTITY FROM FILE NAME
# ============================================================

def identity_from_filename(path):
    stem = path.stem.upper()

    # Examples:
    # AUTH-001.jpg
    # AUTH-001-side.jpg
    # AUTH-002-front.jpg
    # AUTH-002-left.jpg
    # AUTH-002-right.jpg

    m = re.match(
        r"^(.+?)[_-](?:FRONT|SIDE|LEFT|RIGHT|BACK|REAR|PROFILE|ANGLE|VIEW|[0-9]{1,2})$",
        stem
    )

    if m:
        return m.group(1)

    return stem


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):

    h, w = image.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None:
        return []

    valid_faces = []

    for face in faces:

        x, y, fw, fh = [int(v) for v in face[:4]]

        # Ignore impossible boxes
        if fw <= 0 or fh <= 0:
            continue

        # Smaller threshold than your previous 40x40 filter
        if fw < 20 or fh < 20:
            continue

        # Clamp coordinates
        x = max(0, x)
        y = max(0, y)

        if x + fw > w:
            fw = w - x

        if y + fh > h:
            fh = h - y

        if fw <= 0 or fh <= 0:
            continue

        valid_faces.append(face)

    return valid_faces


# ============================================================
# LOAD AUTHORIZED FACE DATABASE
# ============================================================

print()
print("[3] Loading authorized face database...")
print("Path:", FACE_DIR)

database = {}

for image_path in sorted(FACE_DIR.iterdir()):

    if image_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("[ERROR] Cannot read:", image_path.name)
        continue

    faces = detect_faces(image)

    if not faces:
        print("[NO FACE]", image_path.name)
        continue

    # Largest face
    face = max(
        faces,
        key=lambda f: float(f[2] * f[3])
    )

    try:
        aligned = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned
        )

    except Exception as e:
        print("[ERROR] SFace:", image_path.name, e)
        continue

    identity = identity_from_filename(image_path)

    database.setdefault(identity, []).append(
        {
            "filename": image_path.name,
            "feature": feature
        }
    )

    print(
        f"[OK] {image_path.name:25} "
        f"-> {identity}"
    )


print()
print("Database identities:")

for identity, samples in database.items():
    print(
        f"  {identity}: "
        f"{len(samples)} sample(s)"
    )

if not database:
    raise RuntimeError(
        "No authorized faces could be loaded."
    )

# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("[4] Opening video:")
print(VIDEO_PATH)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video."
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = total_frames / fps

print()
print("Video information:")
print("  Resolution :", width, "x", height)
print("  FPS        :", round(fps, 2))
print("  Frames     :", total_frames)
print("  Duration   :", round(duration, 2), "seconds")

# ============================================================
# OUTPUT VIDEO
# ============================================================

output_path = (
    OUTPUT_DIR /
    f"face_test_{VIDEO_PATH.stem}.mp4"
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )

# ============================================================
# STATISTICS
# ============================================================

frame_number = 0
frames_with_faces = 0
total_face_detections = 0

authorized_count = 0
unknown_count = 0

best_score_seen = 0.0

# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("[5] Starting video face test...")
print()
print("Press CTRL+C if you want to stop.")
print()

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        faces = detect_faces(frame)

        if faces:
            frames_with_faces += 1
            total_face_detections += len(faces)

        # ----------------------------------------------------
        # PROCESS EACH FACE
        # ----------------------------------------------------

        for face in faces:

            x, y, fw, fh = [
                int(v) for v in face[:4]
            ]

            try:

                aligned = recognizer.alignCrop(
                    frame,
                    face
                )

                query_feature = recognizer.feature(
                    aligned
                )

            except Exception:
                continue

            best_identity = "UNKNOWN"
            best_score = -1.0
            best_reference = ""

            # Compare against EVERY authorized sample
            for identity, samples in database.items():

                for sample in samples:

                    try:

                        score = float(
                            recognizer.match(
                                query_feature,
                                sample["feature"],
                                cv2.FaceRecognizerSF_FR_COSINE
                            )
                        )

                    except Exception:
                        continue

                    if score > best_score:

                        best_score = score
                        best_identity = identity
                        best_reference = sample["filename"]

            best_score_seen = max(
                best_score_seen,
                best_score
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            if best_score >= FACE_THRESHOLD:

                status = "AUTHORIZED"
                authorized_count += 1

                label = (
                    f"AUTHORIZED - "
                    f"{best_identity} "
                    f"{best_score:.2f}"
                )

                # Green
                box_color = (0, 220, 0)

            else:

                status = "UNKNOWN"
                unknown_count += 1

                label = (
                    f"UNKNOWN "
                    f"{best_score:.2f}"
                )

                # Red
                box_color = (0, 0, 255)

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                box_color,
                3
            )

            # ------------------------------------------------
            # LABEL BACKGROUND
            # ------------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thickness = 2

            (tw, th), baseline = cv2.getTextSize(
                label,
                font,
                scale,
                thickness
            )

            label_y = max(
                25,
                y - 8
            )

            cv2.rectangle(
                frame,
                (x, label_y - th - 10),
                (x + tw + 10, label_y + 3),
                box_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 5, label_y - 3),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # DEBUG INFORMATION
            # ------------------------------------------------

            debug = (
                f"Face {fw}x{fh} | "
                f"Ref: {best_reference}"
            )

            cv2.putText(
                frame,
                debug,
                (10, height - 20),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Console output every recognized face
            if frame_number % 30 == 0:

                print(
                    f"Frame {frame_number:6} | "
                    f"Face {fw:3}x{fh:<3} | "
                    f"{best_identity:<10} | "
                    f"Score={best_score:.3f} | "
                    f"{status}"
                )

        # ----------------------------------------------------
        # FRAME INFORMATION
        # ----------------------------------------------------

        info = (
            f"Frame: {frame_number}/{total_frames} | "
            f"Faces: {len(faces)}"
        )

        cv2.putText(
            frame,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        # Progress
        if frame_number % 100 == 0:

            percent = (
                frame_number /
                max(1, total_frames)
            ) * 100

            print(
                f"Progress: {percent:.1f}%"
            )

except KeyboardInterrupt:

    print()
    print("Stopped by user.")

finally:

    cap.release()
    writer.release()

# ============================================================
# FINAL RESULT
# ============================================================

print()
print("==============================================")
print(" TEST COMPLETE")
print("==============================================")

print()
print("Frames processed       :", frame_number)
print("Frames with faces      :", frames_with_faces)
print("Total face detections  :", total_face_detections)
print("Authorized detections  :", authorized_count)
print("Unknown detections     :", unknown_count)
print("Best SFace score       :", round(best_score_seen, 3))

print()
print("OUTPUT:")
print(output_path)

print()
print("Open the output video and check the face boxes.")
print()