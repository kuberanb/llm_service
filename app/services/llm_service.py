
import asyncio
import logging
import random

from openai import (
    AsyncOpenAI,
    APITimeoutError,
    APIError,
    APIStatusError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIConnectionError,
)

from app.config import settings
from app.schemas.response import (
    UsageInfo,
    GenerateResponse,
    LLMResponse,
)


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=settings.LLM_TIMEOUT,
    max_retries=0,
)

log = logging.getLogger(__name__)


# --------------------------------------------------
# Custom Exception
# --------------------------------------------------

class LLMException(Exception):

    def __init__(self, status_code: int, message: str):

        self.status_code = status_code
        self.message = message

        super().__init__(message)


# --------------------------------------------------
# System Instructions
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant.

Answer the user's questions clearly and accurately.

Return:
1. A detailed answer
2. A short answer
3. Important key points
"""


# --------------------------------------------------
# Helper: Retry Delay
# --------------------------------------------------

async def wait_before_retry(attempt: int):

    delay = 2 ** attempt

    delay += random.uniform(0, 0.5)

    log.info("Retrying in %.2f seconds", delay)

    await asyncio.sleep(delay)


# --------------------------------------------------
# Helper: Extract Token Usage
# --------------------------------------------------

def get_usage(response) -> UsageInfo:

    usage = response.usage

    if usage is None:

        return UsageInfo(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            is_estimated=True,
        )

    return UsageInfo(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        is_estimated=False,
    )


# --------------------------------------------------
# Helper: Call LLM
# --------------------------------------------------

async def call_model(
    messages: list,
    model: str,
    max_output_tokens: int = 1000,
):

    response = await client.responses.parse(
        model=model,
        input=messages
        
        # [
        #     {
        #         "role": "system",
        #         "content": SYSTEM_INSTRUCTIONS,
        #     },
        #     {
        #         "role": "user",
        #         "content": message,
        #     },
        # ]
        
        ,
        text_format=LLMResponse,
        max_output_tokens=max_output_tokens,
    )

    return response


# --------------------------------------------------
# Helper: Stream From Model
# --------------------------------------------------

async def stream_from_model(
    messages: list,
    model: str,
    max_output_tokens: int,
):

    async with client.responses.stream(
        model=model,
        input= messages,
        
        # [
        #     {
        #         "role": "system",
        #         "content": SYSTEM_INSTRUCTIONS,
        #     },
        #     {
        #         "role": "user",
        #         "content": message,
        #     },
        # ],

        text_format=LLMResponse,
        max_output_tokens=max_output_tokens,
    ) as stream:

        async for event in stream:

            if event.type == "response.output_text.delta":

                yield {
                    "type": "delta",
                    "text": event.delta,
                }

        final_response = await stream.get_final_response()

        llm_result = final_response.output_parsed

        if llm_result is None:

            raise LLMException(
                502,
                "The AI returned an empty response.",
            )

        usage = get_usage(final_response)

        yield {
            "type": "completed",
            "response": llm_result.model_dump(),
            "usage": usage.model_dump(),
            "model_used": model,
        }


# ==================================================
# 1. COMPLETE LLM RESPONSE
# ==================================================

async def generate_llm_response(
    messages: list,
    max_output_tokens: int = 1000,
) -> GenerateResponse:

    models = list(dict.fromkeys([
        settings.PRIMARY_MODEL,
        settings.FALLBACK_MODEL,
    ]))

    last_error = None

    # Try primary model, then fallback model
    for model in models:

        for attempt in range(settings.MAX_RETRIES + 1):

            try:

                log.info(
                    "Calling LLM | model=%s | attempt=%s",
                    model,
                    attempt + 1,
                )

                response = await call_model(
                    messages=messages,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )

                llm_result = response.output_parsed

                if llm_result is None:

                    raise LLMException(
                        502,
                        "The AI returned an empty response.",
                    )

                usage = get_usage(response)

                return GenerateResponse(
                    response=llm_result,
                    usage=usage,
                    model_used=model,
                )

            # Rate limit: move to fallback
            except RateLimitError as exc:

                log.warning("Rate limit | model=%s", model)

                last_error = exc

                break

            # Timeout / connection error: retry
            except (APITimeoutError, APIConnectionError) as exc:

                log.warning(
                    "Connection error | model=%s | attempt=%s",
                    model,
                    attempt + 1,
                )

                last_error = exc

                if attempt < settings.MAX_RETRIES:

                    await wait_before_retry(attempt)
                    continue

                break

            # Server errors: retry
            except APIStatusError as exc:

                log.exception("API status error: %s", exc.status_code)

                last_error = exc

                if exc.status_code >= 500:

                    if attempt < settings.MAX_RETRIES:

                        await wait_before_retry(attempt)
                        continue

                    break

                raise LLMException(
                    exc.status_code,
                    "The AI request could not be processed.",
                )

            except AuthenticationError:

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

            except LLMException:

                raise

            except APIError as exc:

                log.exception("Unexpected API error")

                last_error = exc
                break

            except Exception:

                log.exception("Unexpected LLM error")

                raise LLMException(
                    500,
                    "Something went wrong while generating the response.",
                )

    # Both models failed
    if isinstance(last_error, RateLimitError):

        raise LLMException(
            429,
            "AI service is currently busy.",
        )

    if isinstance(last_error, (APITimeoutError, APIConnectionError)):

        raise LLMException(
            504,
            "AI service is temporarily unavailable.",
        )

    raise LLMException(
        502,
        "AI service encountered an error.",
    )


# ==================================================
# 2. STREAMING LLM RESPONSE
# ==================================================

async def stream_llm_response(
    messages: list,
    max_output_tokens: int = 1000,
):

    models = list(dict.fromkeys([
        settings.PRIMARY_MODEL,
        settings.FALLBACK_MODEL,
    ]))

    has_sent_delta = False
    last_error = None

    # Try primary model, then fallback model
    for model in models:

        for attempt in range(settings.MAX_RETRIES + 1):

            try:

                log.info(
                    "Starting stream | model=%s | attempt=%s",
                    model,
                    attempt + 1,
                )

                async for event in stream_from_model(
                    messages=messages,
                    model=model,
                    max_output_tokens=max_output_tokens,
                ):

                    if event["type"] == "delta":

                        has_sent_delta = True

                    yield event

                    if event["type"] == "completed":
                        return

                return

            # --------------------------------------
            # Rate limit: move to fallback
            # --------------------------------------

            except RateLimitError as exc:

                log.warning("Rate limit | model=%s", model)

                last_error = exc

                if has_sent_delta:
                    break

                # Stop retrying this model
                break

            # --------------------------------------
            # Timeout / connection error
            # --------------------------------------

            except (APITimeoutError, APIConnectionError) as exc:

                log.warning(
                    "Stream connection error | model=%s | attempt=%s",
                    model,
                    attempt + 1,
                )

                last_error = exc

                if has_sent_delta:
                    break

                if attempt < settings.MAX_RETRIES:

                    await wait_before_retry(attempt)
                    continue

                break

            # --------------------------------------
            # API status error
            # --------------------------------------

            except APIStatusError as exc:

                log.exception(
                    "Streaming API error: %s",
                    exc.status_code,
                )

                last_error = exc

                if has_sent_delta:
                    break

                if exc.status_code >= 500:

                    if attempt < settings.MAX_RETRIES:

                        await wait_before_retry(attempt)
                        continue

                    break

                yield {
                    "type": "error",
                    "error": "api_error",
                    "message": "The AI request could not be processed.",
                    "retryable": False,
                }

                return

            # --------------------------------------
            # Authentication error
            # --------------------------------------

            except AuthenticationError:

                log.exception("Streaming authentication failed")

                yield {
                    "type": "error",
                    "error": "authentication_error",
                    "message": "AI service authentication failed.",
                    "retryable": False,
                }

                return

            # --------------------------------------
            # Invalid request
            # --------------------------------------

            except BadRequestError:

                log.exception("Invalid streaming request")

                yield {
                    "type": "error",
                    "error": "invalid_request",
                    "message": "The AI request could not be processed.",
                    "retryable": False,
                }

                return

            # --------------------------------------
            # Custom LLM error
            # --------------------------------------

            except LLMException as exc:

                log.exception("LLM streaming exception")

                yield {
                    "type": "error",
                    "error": "llm_error",
                    "message": exc.message,
                    "retryable": False,
                }

                return

            # --------------------------------------
            # Other API errors
            # --------------------------------------

            except APIError as exc:

                log.exception("Unexpected streaming API error")

                last_error = exc
                break

            # --------------------------------------
            # Unexpected errors
            # --------------------------------------

            except Exception:

                log.exception("Unexpected streaming error")

                yield {
                    "type": "error",
                    "error": "internal_error",
                    "message": "Something went wrong while generating the response.",
                    "retryable": False,
                }

                return

        # If any text was already sent, never restart
        if has_sent_delta:

            yield {
                "type": "error",
                "error": "stream_interrupted",
                "message": "Response was interrupted. Please try again.",
                "retryable": False,
            }

            return

        # Otherwise, outer loop tries fallback model

    # Both models failed
    if isinstance(last_error, RateLimitError):

        yield {
            "type": "error",
            "error": "rate_limit",
            "message": "Both AI models are currently busy.",
            "retryable": True,
        }

    else:

        yield {
            "type": "error",
            "error": "service_unavailable",
            "message": "AI service is temporarily unavailable.",
            "retryable": True,
        }
