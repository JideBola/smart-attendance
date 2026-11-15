from fastapi import APIRouter

from app.schemas.attendance import AttendanceResponse, HealthResponse

router = APIRouter(tags=['attendance'])


@router.get('/', response_model=HealthResponse)
def home() -> dict[str, str]:
    return {'message': 'Smart attendance API is running'}


@router.get('/attendance', response_model=AttendanceResponse)
def attendance_status() -> dict[str, str]:
    return {'status': 'Attendance route is ready'}
