"""
chatbot_logic.py
The 'brain' of RuleBot - same logic as Project 1's terminal version,
just wrapped in a function so both the terminal app AND the Flask
web server can reuse it without duplicating code.
"""

responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just lines of code, but I'm running smoothly!",
    "what is your name": "I'm RuleBot, a simple rule-based chatbot.",
    "what can you do": "I can respond to a few basic greetings and questions. Try 'help' to see more.",
    "help": "Try saying: hello, how are you, what is your name, what can you do, or bye.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

exit_commands = {"bye", "exit", "quit", "goodbye"}

DEFAULT_REPLY = "I do not understand that yet. Try 'help' to see what I know."


def process_message(raw_text: str) -> dict:
    """
    Takes raw user text, runs it through the same IPO pipeline as the
    terminal bot, and returns a dictionary describing every step -
    this is what the API sends back to the browser as JSON.
    """
    raw_text = raw_text or ""
    clean = raw_text.lower().strip()

    if clean in exit_commands:
        return {
            "raw": raw_text,
            "clean": clean,
            "matched": True,
            "is_exit": True,
            "reply": "Goodbye! Have a great day.",
        }

    matched = clean in responses
    reply = responses.get(clean, DEFAULT_REPLY)

    return {
        "raw": raw_text,
        "clean": clean,
        "matched": matched,
        "is_exit": False,
        "reply": reply,
    }
