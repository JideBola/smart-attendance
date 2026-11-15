from pydantic import BaseModel


class HealthResponse(BaseModel):
    message: str


class AttendanceResponse(BaseModel):
    status: str
