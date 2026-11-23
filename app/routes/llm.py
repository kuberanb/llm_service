
from fastapi import APIRouter
import logging
from app.schemas.request import GenerateRequest
import uuid
import json 
from app.services.llm_service import stream_llm_response,generate_llm_response
from fastapi.responses import StreamingResponse,JSONResponse
from app.services.llm_service import LLMException
from app.schemas.error_response import ErrorResponse


llm_router = APIRouter()

log = logging.getLogger(__name__)


# --------------------------------------------------
# GENERATE ENDPOINT
# --------------------------------------------------

llm_router.post("/generate")
async def generate(request: GenerateRequest):

    request_id = str(uuid.uuid4())

    conversation_id = (
        request.conversation_id
        or str(uuid.uuid4())
    )

    if request.stream:

        async def response_generator():

        # Start event

         start_event = {
                "type": "start",
                "request_id": request_id,
                "conversation_id": conversation_id,
            }
         
        yield f"data: {json.dumps(start_event)}\n\n"

        # Stream LLM response
        async for event in stream_llm_response(
           message= request.query,
           max_output_tokens = request.max_output_tokens
        ):
            yield f"data: {json.dumps(event)}\n\n"

        # End event
        yield "data: [DONE]\n\n"

        return StreamingResponse(
          response_generator(),
          media_type="text/event-stream",
          headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
          }

       )
    else:

# --------------------------------------------------
# COMPLETE RESPONSE
# --------------------------------------------------
    
        try:
           

           response = await generate_llm_response(message=request.query,       
                max_output_tokens=request.max_output_tokens,
             )

           return response


        except LLMException as exc:

            log.error(
            "LLM request failed. Request ID: %s",
            request_id,
        )

        error_response = ErrorResponse(
            request_id=request_id,
            error="llm_error",
            message=exc.message,
            retryable=exc.status_code in (429, 502, 504),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
        )
