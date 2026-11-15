# Smart Attendance API

FastAPI backend plus local OpenCV scripts for enrolling and recognizing faces.

## Run the API

```bash
./venv/bin/uvicorn main:app --reload
```

Useful endpoints:

- `GET /`
- `GET /attendance`
- `GET /face/status`
- `GET /face/dataset`

## Face Scripts

Enroll a face:

```bash
./venv/bin/python scripts/enroll_face.py
```

Recognize enrolled faces:

```bash
./venv/bin/python scripts/recognize_face.py
```

Face data is stored locally in `data/*.pkl`. Treat those files as runtime data,
not source code.
# smart-attendance
