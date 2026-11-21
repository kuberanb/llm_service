
import tiktoken

def count_tokens(text:str, model: str = "gpt-4.1-mini")->int :
    try:

        encoding = tiktoken.encoding_for_model(model)

    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

def count_message_tokens(messages:list[dict], model: str = "gpt-4.1-mini" )->int :

    total_tokens = 0

    for message in messages:
        total_tokens += count_tokens(   
            message.get("content", ""),
            model
)

        return total_tokens