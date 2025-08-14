import signal

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def get_input_with_timeout(prompt, timeout=10, default=None):
    """
    Prompts the user for input and uses a default value if no input is received within the timeout.

    Parameters:
        prompt (str): The message displayed to the user.
        timeout (int): The number of seconds to wait for user input.
        default: The default value to return if the user doesn't respond.

    Returns:
        str: The user's input or the default value if no input was received.
    """
    # Set the signal handler for the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the timer

    try:
        user_input = input(prompt)
        signal.alarm(0)  # Disable the alarm
        return user_input if user_input.strip() else default
    except TimeoutException:
        print(f"\nTimeout! Using default value: {default}")
        return default
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled


if __name__ == "__main__":
    prompt = "Please enter your username"
    default = "user"
    get_input_with_timeout(prompt=prompt, timeout=10, default=default)
