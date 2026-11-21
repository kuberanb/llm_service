

from pydantic import BaseModel

class LLMResponse(BaseModel):
    answer : str
    summary : str
    key_points: list[str]


class UsageInfo(BaseModel):

    input_tokens: int
    output_tokens: int 
    total_tokens: int
    estimated_cost:float


class GenerateResponse(BaseModel):
    reponse: LLMResponse
    usage: UsageInfo
    model_used : str