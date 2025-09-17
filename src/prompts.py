"""
Prompt definitions for image description using the visual language model (VLM).

This module contains helper functions that return text prompts used
by the ``GeminiVLM`` to guide image descriptions. Keeping prompts
centralised makes it easier to adjust the behaviour of the model
without scattering strings throughout the codebase.
"""

def default_prompt() -> str:
    """Return the default prompt for describing images in Bahasa Indonesia.

    The prompt instructs the VLM to act as a sighted assistant for
    visually impaired readers. It provides guidelines on how to
    describe logos, diagrams, charts and tables. See the upstream
    implementation for more details.

    :return: A multi‑line string used as a prompt.
    """
    return (
        "You are describing an image from a document for someone who cannot see it.\n"
        "Your response will be embedded to the document so don't add anything extra.\n"
        "ALWAYS answer in Bahasa Indonesia\n"
        "- Start with a short title.\n"
        "- If it is a logo, ignore it and say 'Logo'.\n"
        "- If it is a flow/diagram: list steps in order; include decisions/branches.\n"
        "- If it is a chart: mention axes, units, legend, series, trends, notable values.\n"
        "- If it is a table-like figure: summarise key fields as bullets (not CSV).\n"
        "Be concise, factual, and faithful."
    )