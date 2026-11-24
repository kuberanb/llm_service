
from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):

    conversation_id: Optional[str] = None

    query : str = Field(
        # min_length=3,
        # max_length=5000
    )

    max_output_tokens: int = Field(
        # default=500,
        # ge=50,
        # le=2000
    )
    stream: bool = False
    