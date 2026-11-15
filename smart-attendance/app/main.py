from fastapi import FastAPI

from app.routes.attendance import router as attendance_router
from app.routes.face import router as face_router

app = FastAPI(title='Smart Attendance API')

app.include_router(attendance_router)
app.include_router(face_router)
