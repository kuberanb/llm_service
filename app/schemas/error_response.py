
from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):

    request_id: str

    error: str

    message: str

    retryable: bool