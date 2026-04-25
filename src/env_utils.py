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
    Private helper that handles the common logic for retrieving and parsing 
    environment variables. It manages whitespace stripping, empty checks,
    and error logging.
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
    Safely retrieves and parses an integer from the system's environment variables.
    
    This function handles cases where the variable might be missing, empty, 
    or contain non-numeric characters. In such cases, it logs a warning 
    and returns a safe default value to prevent the application from crashing.

    Args:
        environment_variable_name: The exact name of the environment variable to look up.
        fallback_value: The integer value to return if lookup or parsing fails.

    Returns:
        The parsed integer from the environment, or the fallback_value on failure.
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        int, 
        "integer"
    )

def get_float_from_environment(environment_variable_name: str, fallback_value: float) -> float:
    """
    Safely retrieves and parses a decimal (float) from the system's environment variables.
    
    Similar to the integer version, this ensures that malformed decimal configurations 
    do not stop the audio processing system.

    Args:
        environment_variable_name: The exact name of the environment variable to look up.
        fallback_value: The float value to return if lookup or parsing fails.

    Returns:
        The parsed float from the environment, or the fallback_value on failure.
    """
    return _get_parsed_value_from_environment(
        environment_variable_name, 
        fallback_value, 
        float, 
        "decimal"
    )
