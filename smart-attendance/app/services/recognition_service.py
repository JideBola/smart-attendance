from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier

from app.services.face_service import (
    face_to_vector,
    load_face_dataset,
    load_face_detector,
    open_camera,
    require_cv2,
)


def train_face_classifier(n_neighbors: int = 5) -> KNeighborsClassifier:
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be greater than zero.")

    dataset = load_face_dataset()

    if len(dataset.names) == 0:
        raise RuntimeError("No enrolled face samples found.")

    effective_neighbors = min(n_neighbors, len(dataset.names))
    classifier = KNeighborsClassifier(n_neighbors=effective_neighbors)
    classifier.fit(dataset.faces, dataset.names)
    return classifier


def annotate_frame(
    frame: Any,
    detected_faces: Any,
    classifier: KNeighborsClassifier,
) -> Any:
    cv2_module = require_cv2()

    for x, y, w, h in detected_faces:
        crop_img = frame[y : y + h, x : x + w, :]
        predicted_name = classifier.predict(face_to_vector(crop_img))[0]

        cv2_module.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2_module.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2_module.putText(
            frame,
            f"Hello {predicted_name}",
            (x, y - 10),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def recognize_faces(
    *,
    camera_index: int = 0,
    n_neighbors: int = 5,
) -> None:
    cv2_module = require_cv2()
    classifier = train_face_classifier(n_neighbors=n_neighbors)
    face_detector = load_face_detector()
    video = open_camera(camera_index)

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
            faces_in_frame = face_detector.detectMultiScale(gray, 1.3, 5)
            annotate_frame(
                frame,
                faces_in_frame,
                classifier,
            )

            cv2_module.imshow("Recognition", frame)
            if cv2_module.waitKey(1) == ord("q"):
                break
    finally:
        video.release()
        cv2_module.destroyAllWindows()
