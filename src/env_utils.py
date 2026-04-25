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
    Internal helper to centralize environment variable parsing logic.
    ──────────────────────────────────────────────────────────────────
    ARGUMENTS:
      environment_variable_name : The key to find in the OS environment.
      fallback_value           : Default numeric value returned on any failure.
      parser_function          : The type constructor (int or float).
      type_label               : Description of the type for log messages.
    ──────────────────────────────────────────────────────────────────
    INTERNAL LOGIC:
      1. Fetches the raw value from the environment via os.environ.
      2. Trims whitespace and checks if the value is empty.
      3. Attempts to call the parser_function on the string.
      4. Catches ValueError, logs a warning, and returns the fallback.
    ──────────────────────────────────────────────────────────────────
    RETURNS: T (The parsed numeric value or the fallback)
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
    Retrieve an integer from the environment without risking a crash.
    ──────────────────────────────────────────────────────────────────
    ARGUMENTS:
      environment_variable_name : The key to find in the OS environment.
      fallback_value           : Default integer if key is missing/bad.
    ──────────────────────────────────────────────────────────────────
    INTERNAL LOGIC:
      The function first removes whitespace using .strip(). If the 
      remaining text is empty, it returns the fallback. It then tries 
      to convert the text to an integer. If the text is invalid (like 
      '10.5' or 'abc'), it logs a warning and returns the fallback.
    ──────────────────────────────────────────────────────────────────
    RETURNS: Integer
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        int, 
        "integer"
    )

def get_float_from_environment(environment_variable_name: str, fallback_value: float) -> float:
    """
    Retrieve a decimal (float) from the environment without risking a crash.
    ──────────────────────────────────────────────────────────────────
    ARGUMENTS:
      environment_variable_name : The key to find in the OS environment.
      fallback_value           : Default float if key is missing/bad.
    ──────────────────────────────────────────────────────────────────
    INTERNAL LOGIC:
      This function is a safe wrapper around the float() constructor.
      It strips whitespace and handles empty or non-numeric strings by 
      returning the provided fallback value and logging a warning.
    ──────────────────────────────────────────────────────────────────
    RETURNS: Float
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        float, 
        "decimal"
    )
