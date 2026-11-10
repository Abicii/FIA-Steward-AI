import os

from llama_cloud import LlamaCloud


class LlamaParseConfigError(RuntimeError):
    """Raised when LlamaParse cannot be configured from the environment."""


def parse_pdf(path: str) -> str:
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise LlamaParseConfigError("LLAMA_CLOUD_API_KEY is not set")

    client = LlamaCloud()
    uploaded = client.files.create(file=path, purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded.id,
        tier="fast",
        version="latest",
        expand=["markdown"],
        crop_box={
            "top": 0.08,
            "bottom": 0.08,
            "left": 0.0,
            "right": 0.0,
        },
    )

    pages = getattr(getattr(result, "markdown", None), "pages", None) or []
    return "\n\n".join(
        page.markdown
        for page in pages
        if getattr(page, "success", True) and getattr(page, "markdown", None)
    )
