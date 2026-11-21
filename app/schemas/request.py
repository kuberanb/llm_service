
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):

    query : str = Field(
        min_length=3,
        max_length=5000
    )

    max_output_tokens: str = Field(
        default=500,
        ge=50,
        le=2000
    )

    