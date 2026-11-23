

import asyncio
import json
import logging
from pydantic import ValidationError
from openai import AsyncOpenAI,APITimeoutError,APIError,RateLimitError,AuthenticationError,BadRequestError,APIError
from app.config import settings
from app.schemas.response import UsageInfo
from app.schemas.response import GenerateResponse
from app.schemas.response import LLMResponse

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=settings.LLM_TIMEOUT,
    max_retries=0,
)

log = logging.getLogger(__name__)

class LLMException(Exception):
    def __init__(self, status_code:int ,message:str):
        self.status_code = status_code
        self.message = message


SYSTEM_INSTRUCTIONS = """ 
You are a helpful AI assistant.

Answer the user's questions clearly and accurately.

Return:
1. A detailed answer
2. A short answer
3. Important key points 
"""


# --------------------------------------------------
# Helper: Extract Token Usage
# --------------------------------------------------

def get_usage(response) -> UsageInfo:

    usage: UsageInfo = response.usage

    if usage is None:
        return UsageInfo(input_tokens=0,output_tokens=0,total_tokens=0,is_estimated=True)

    return UsageInfo(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        is_estimated=False
    )




# --------------------------------------------------
# 1. COMPLETE LLM RESPONSE
# --------------------------------------------------

async def generate_llm_response(message:str,max_output_tokens:int = 1000,)-> GenerateResponse:

    try:

        response = await client.responses.parse(
            model=settings.PRIMARY_MODEL,
            input=[
                {
                    "role":"system",
                    "content":SYSTEM_INSTRUCTIONS
                },
                {
                    "role":"user",
                    "content":message
                }
            ],
            text_format=LLMResponse,
            max_output_tokens=max_output_tokens,
        )

        # Get validated Pydantic response
        llm_result = response.output_parsed

        if llm_result is None:
            raise LLMException(502,"The AI returned an empty response")

        # Extract token usage 
        usage = get_usage(response)

        # Return complete API response
        return GenerateResponse(response=llm_result,usage=usage,)

    except APITimeoutError :

        log.exception("LLM request timeout")

        raise LLMException(status_code=504,message="The AI service took too long to respond",
                           )
    except RateLimitError:
        log.exception("LLM rate limit exceeded")

        raise LLMException(status_code=429, message="The Ai service is currently busy",
                           )
    
    except AuthenticationError :

        log.exception("LLM authentication failed")

        raise LLMException(
            500,
            "AI service authentication failed.",
        )

    except BadRequestError:

        log.exception("Invalid LLM request")

        raise LLMException(
            400,
            "The AI request could not be processed.",
        )

    except APIError:

        log.exception("OpenAI API error")

        raise LLMException(
            502,
            "The AI service encountered an error.",
        )

    except LLMException:
        raise

    except Exception:

        log.exception("Unexpected LLM error")

        raise LLMException(status_code=500, message="Something went wrong while generating the response",)



# --------------------------------------------------
# 2. STREAMING LLM RESPONSE
# --------------------------------------------------

async def stream_llm_response(
        message:str,
        max_output_tokens:int = 1000,
):
    try:

        async with client.responses.stream(
            model= settings.PRIMARY_MODEL,
            input=[
                {
                    "role":"system",
                    "content":SYSTEM_INSTRUCTIONS
                }
            ]
        ) as stream :
            async for event in stream:

                if type == "response.output_text.delta":
                    yield {
                        "type": "delta",
                        "text":event.delta
                    }

            # Get final completed response
            final_response = await stream.get_final_response()

            llm_result = final_response.output_parsed

            if llm_result is None:
                raise LLMException(502,"The AI returned an empty response")

            usage = get_usage(final_response)

            # Send complete validated response

            yield{
                "type": "completed",
                "response":llm_result.model_dump(),
            }

    except APITimeoutError:

        log.exception("Streaming request timed out")

        yield {
            "type": "error",
            "error": "timeout",
            "message": "The AI service took too long to respond.",
            "retryable": True,
        }

    except RateLimitError:

        log.exception("Streaming rate limit exceeded")

        yield {
            "type": "error",
            "error": "rate_limit",
            "message": "The AI service is currently busy.",
            "retryable": True,
        }

    except AuthenticationError:

        log.exception("Streaming authentication failed")

        yield {
            "type": "error",
            "error": "authentication_error",
            "message": "AI service authentication failed.",
            "retryable": False,
        }

    except BadRequestError:

        log.exception("Invalid streaming request")

        yield {
            "type": "error",
            "error": "invalid_request",
            "message": "The AI request could not be processed.",
            "retryable": False,
        }

    except APIError:

        log.exception("Streaming API error")

        yield {
            "type": "error",
            "error": "api_error",
            "message": "The AI service encountered an error.",
            "retryable": True,
        }

    except Exception:

        log.exception("Unexpected streaming error")

        yield {
            "type": "error",
            "error": "internal_error",
            "message": "Something went wrong while generating the response.",
            "retryable": False,
        }

    except:
        pass   