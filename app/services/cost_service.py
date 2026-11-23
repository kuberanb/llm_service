import json

from pathlib import Path


PRICING_FILE = Path(__file__).parent / "pricing.json"


with open(PRICING_FILE, "r") as file:

    MODEL_PRICING = json.load(file)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0
) -> float:

    pricing = MODEL_PRICING.get(model)

    if pricing is None:

        raise ValueError(
            f"Pricing not available for model: {model}"
        )

    regular_input_tokens = (
        input_tokens - cached_tokens
    )

    input_cost = (
        regular_input_tokens
        * pricing["input"]
        / 1_000_000
    )

    cached_input_cost = (
        cached_tokens
        * pricing["cached_input"]
        / 1_000_000
    )

    output_cost = (
        output_tokens
        * pricing["output"]
        / 1_000_000
    )

    total_cost = (
        input_cost
        + cached_input_cost
        + output_cost
    )

    return round(total_cost, 8)
