from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SAMPLES = 100
SAMPLE_EVERY_N_FRAMES = 10
FACE_IMAGE_SIZE = (50, 50)
DATA_DIR = PROJECT_ROOT / "data"
NAMES_FILE = DATA_DIR / "names.pkl"
FACES_FILE = DATA_DIR / "faces_data.pkl"


@dataclass(frozen=True)
class FaceDataset:
    names: list[str]
    faces: np.ndarray


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install opencv-python first.")
    return cv2


def load_face_detector() -> Any:
    cv2_module = require_cv2()
    detector = cv2_module.CascadeClassifier(
        cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Failed to load the Haar cascade classifier.")
    return detector


def open_camera(camera_index: int = 0) -> Any:
    cv2_module = require_cv2()
    video = cv2_module.VideoCapture(camera_index)
    if not video.isOpened():
        video.release()
        raise RuntimeError(f"Failed to open camera {camera_index}.")
    return video


def normalize_name(name: str) -> str:
    normalized = " ".join(name.strip().split())
    if not normalized:
        raise ValueError("Name is required to save face data.")
    return normalized


def face_to_vector(face_image: np.ndarray) -> np.ndarray:
    cv2_module = require_cv2()
    resized_img = cv2_module.resize(face_image, FACE_IMAGE_SIZE)
    return resized_img.astype("float32").reshape(1, -1)


def load_face_dataset() -> FaceDataset:
    if not NAMES_FILE.exists() and not FACES_FILE.exists():
        raise FileNotFoundError("No enrolled face data found. Enroll a face first.")

    if not NAMES_FILE.exists() or not FACES_FILE.exists():
        raise RuntimeError("Face data files are incomplete.")

    with NAMES_FILE.open("rb") as file:
        names = pickle.load(file)

    with FACES_FILE.open("rb") as file:
        faces = pickle.load(file)

    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RuntimeError("Stored names data is invalid.")

    faces_array = np.asarray(faces, dtype="float32")
    if faces_array.ndim != 2:
        raise RuntimeError("Stored face data must be a 2D feature matrix.")

    if len(names) != len(faces_array):
        raise RuntimeError("Stored names and face samples are out of sync.")

    return FaceDataset(names=names, faces=faces_array)


def save_face_samples(name: str, samples: np.ndarray) -> int:
    normalized_name = normalize_name(name)
    samples_array = np.asarray(samples, dtype="float32")

    if samples_array.ndim != 2 or len(samples_array) == 0:
        raise ValueError("Face samples must be a non-empty 2D matrix.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if NAMES_FILE.exists() != FACES_FILE.exists():
        raise RuntimeError("Face data files are incomplete. Fix or remove them first.")

    if NAMES_FILE.exists():
        dataset = load_face_dataset()
        if dataset.faces.shape[1] != samples_array.shape[1]:
            raise RuntimeError("New face samples do not match stored sample dimensions.")
        names = [*dataset.names, *([normalized_name] * len(samples_array))]
        faces = np.concatenate([dataset.faces, samples_array], axis=0)
    else:
        names = [normalized_name] * len(samples_array)
        faces = samples_array

    with NAMES_FILE.open("wb") as file:
        pickle.dump(names, file)

    with FACES_FILE.open("wb") as file:
        pickle.dump(faces, file)

    return len(samples_array)


def dataset_status() -> dict[str, str | int | bool]:
    if not NAMES_FILE.exists() or not FACES_FILE.exists():
        return {
            "available": False,
            "samples": 0,
            "enrolled_people": 0,
            "names_file": str(NAMES_FILE),
            "faces_file": str(FACES_FILE),
            "message": "No enrolled face data found.",
        }

    try:
        dataset = load_face_dataset()
    except Exception as exc:
        return {
            "available": False,
            "samples": 0,
            "enrolled_people": 0,
            "names_file": str(NAMES_FILE),
            "faces_file": str(FACES_FILE),
            "message": str(exc),
        }

    return {
        "available": True,
        "samples": len(dataset.names),
        "enrolled_people": len(set(dataset.names)),
        "names_file": str(NAMES_FILE),
        "faces_file": str(FACES_FILE),
        "message": "Face dataset is available.",
    }


def collect_face_samples(
    *,
    target_samples: int = TARGET_SAMPLES,
    sample_every_n_frames: int = SAMPLE_EVERY_N_FRAMES,
    camera_index: int = 0,
) -> np.ndarray:
    if target_samples <= 0:
        raise ValueError("target_samples must be greater than zero.")

    if sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be greater than zero.")

    cv2_module = require_cv2()
    detector = load_face_detector()
    video = open_camera(camera_index)

    samples: list[np.ndarray] = []
    frame_count = 0

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
            detected_faces = detector.detectMultiScale(gray, 1.3, 5)

            if len(detected_faces) > 0:
                x, y, w, h = max(detected_faces, key=lambda box: box[2] * box[3])
                crop_img = frame[y : y + h, x : x + w, :]

                if frame_count % sample_every_n_frames == 0:
                    samples.append(face_to_vector(crop_img).reshape(-1))

                cv2_module.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            frame_count += 1
            cv2_module.putText(
                frame,
                str(len(samples)),
                (50, 50),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                1,
                (50, 50, 255),
                1,
            )
            cv2_module.imshow("Enroll face", frame)

            key = cv2_module.waitKey(1)
            if key == ord("q") or len(samples) >= target_samples:
                break
    finally:
        video.release()
        cv2_module.destroyAllWindows()

    return np.asarray(samples, dtype="float32")


def run_camera(name: str | None = None) -> None:
    if name is None:
        name = input("Enter your name: ")

    normalized_name = normalize_name(name)
    samples = collect_face_samples()

    if len(samples) < TARGET_SAMPLES:
        print(
            f"Collected {len(samples)} samples. "
            f"Capture {TARGET_SAMPLES} samples before saving."
        )
        return

    saved_count = save_face_samples(normalized_name, samples)
    print(f"Saved {saved_count} face samples for {normalized_name}.")


def camera_status(camera_index: int = 0) -> dict[str, str | bool]:
    if cv2 is None:
        return {
            "available": False,
            "message": "OpenCV is not installed. Install opencv-python to enable camera access.",
        }

    video = cv2.VideoCapture(camera_index)
    is_open = video.isOpened()
    video.release()

    if is_open:
        return {"available": True, "message": "Camera is available."}

    return {"available": False, "message": "Camera is not available."}


if __name__ == "__main__":
    run_camera()
