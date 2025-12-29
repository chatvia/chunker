from pathlib import Path


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found at: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

