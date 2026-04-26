import os
import logging
from typing import TypeVar, Callable

# Set up logging to track when configuration fallbacks occur.
# This helps developers identify malformed environment variables in production.
logger = logging.getLogger(__name__)

# Define a generic TypeVar for numeric types (int or float)
T = TypeVar("T", int, float)

def _get_parsed_value_from_environment(
    environment_variable_name: str, 
    fallback_value: T, 
    parser_function: Callable[[str], T],
    type_label: str
) -> T:
    """
    This is an internal helper that safely gets a number from the computer's environment settings.

    It handles the logic of finding the value, cleaning it, and turning it into a number
    without letting the program crash if the value is wrong.

    Step-by-step:
    1. Look for the setting using the name provided (the environment variable).
    2. Remove any extra spaces at the beginning or end of the value.
    3. If the value is empty or doesn't exist, return the default fallback value.
    4. Try to turn the text into a number (either an integer or a decimal).
    5. If the text isn't a valid number, log a warning message so the developer knows.
    6. Return the fallback value if anything went wrong, otherwise return the new number.
    """
    # Retrieve the value and remove any accidental leading or trailing whitespace.
    raw_environment_value = os.environ.get(environment_variable_name, "").strip()
    
    # If the value is missing or just whitespace, use the provided fallback.
    if not raw_environment_value:
        return fallback_value
        
    try:
        # Attempt to convert the cleaned string value using the provided parser function.
        return parser_function(raw_environment_value)
    except ValueError:
        # If parsing fails, log a clear warning and return the safe fallback.
        logger.warning(
            f"Configuration Error: '{environment_variable_name}' has an invalid {type_label} "
            f"value: '{raw_environment_value}'. Falling back to default: {fallback_value}."
        )
        return fallback_value

def get_integer_from_environment(environment_variable_name: str, fallback_value: int) -> int:
    """
    This function looks for a whole number (integer) in your environment settings.

    It ensures that if the setting is missing or contains text that isn't a number,
    your program keeps running smoothly by using a safe default value.

    Step-by-step:
    1. Ask the internal helper to find the value and try to turn it into an integer.
    2. If it works, you get the number from your settings.
    3. If it fails or is missing, you get the fallback value instead.
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        int, 
        "integer"
    )

def get_float_from_environment(environment_variable_name: str, fallback_value: float) -> float:
    """
    This function looks for a decimal number (float) in your environment settings.

    It works just like the integer version but allows for numbers with dots (like 10.5).
    It prevents crashes by returning a default value if the setting is broken or empty.

    Step-by-step:
    1. Ask the internal helper to find the value and try to turn it into a decimal number.
    2. If it works, you get the decimal number from your settings.
    3. If it fails or is missing, you get the decimal number from your settings.
    4. If it fails or is missing, you get the fallback value instead.
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        float, 
        "decimal"
    )
