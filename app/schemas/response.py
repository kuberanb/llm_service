from pydantic import BaseModel, Field, ConfigDict


class LLMResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    answer: str

    summary: str

    key_points: list[str] = Field(
        min_length=1
    )


class UsageInfo(BaseModel):

    input_tokens: int = Field(ge=0)

    output_tokens: int = Field(ge=0)

    total_tokens: int = Field(ge=0)

    is_estimated: bool = False


class GenerateResponse(BaseModel):

    response: LLMResponse

    usage: UsageInfo

    model_used: str