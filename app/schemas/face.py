from pydantic import BaseModel


class CameraStatusResponse(BaseModel):
    available: bool
    message: str


class FaceDatasetStatusResponse(BaseModel):
    available: bool
    samples: int
    enrolled_people: int
    names_file: str
    faces_file: str
    message: str
