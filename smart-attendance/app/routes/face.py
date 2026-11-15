from fastapi import APIRouter

from app.schemas.face import CameraStatusResponse, FaceDatasetStatusResponse
from app.services.face_service import camera_status, dataset_status

router = APIRouter(prefix='/face', tags=['face'])


@router.get('/status', response_model=CameraStatusResponse)
def face_status() -> dict[str, str | bool]:
    return camera_status()


@router.get('/dataset', response_model=FaceDatasetStatusResponse)
def face_dataset_status() -> dict[str, str | int | bool]:
    return dataset_status()
