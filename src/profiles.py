from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str
    rag_template: str
    chat_template: str
    translate_template: str
    stop_rag: list[str] = field(default_factory=list)
    stop_chat: list[str] = field(default_factory=list)
    stop_translate: list[str] = field(default_factory=list)
    temperature: float = 0.3
    repeat_penalty: float = 1.3
    translate_repeat_penalty: float = 1.5
    chat_format: str = "raw"
    n_ctx_default: int = 4096
    patches: list[str] = field(default_factory=list)
    use_chat_api: bool = False  # If True, use create_chat_completion instead of create_completion


AYA = ModelProfile(
    name="aya",
    rag_template=(
        "You are a helpful assistant at a community knowledge kiosk.\n\n"
        "Rules:\n"
        "- ONLY use information from the Context below\n"
        '- If the Context does not answer the question, reply ONLY with: "I don\'t have information about that."\n'
        "- Do NOT make up information\n"
        "- Do NOT add information from your own knowledge\n"
        "- Answer in the same language as the question\n"
        "- Keep your answer short and direct\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    chat_template="{system}\n\nUser: {message}\n\nAssistant:",
    translate_template="Translate from {source} to {target}.\n\n{source}: {text}\n\n{target}:",
    stop_rag=["\nQuestion:", "\n\n\n", "\nNote:", "(Note:", "\nAnswer:"],
    stop_chat=["\nUser:", "\n\n\n"],
    stop_translate=[],
    temperature=0.3,
    repeat_penalty=1.3,
    translate_repeat_penalty=1.5,
    chat_format="raw",
    n_ctx_default=4096,
    patches=["jinja2_loopcontrols"],
)

GEMMA = ModelProfile(
    name="gemma",
    rag_template=(
        "You are a helpful assistant.\n\n"
        "Rules:\n"
        "- ONLY use information from the Context below\n"
        '- If the Context does not answer the question, reply ONLY with: "I don\'t have information about that."\n'
        "- Do NOT make up information\n"
        "- Do NOT add information from your own knowledge\n"
        "- Answer in the same language as the question\n"
        "- Keep your answer short and direct\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    chat_template="{system}\nUser: {message}\nAssistant:",
    translate_template="Translate the following from {source} to {target}. Reply with ONLY the translation.\n{source}: {text}\n{target}:",
    stop_rag=["\nQuestion:", "\n\n\n", "\nAnswer:"],
    stop_chat=["\nUser:", "\n\n\n"],
    stop_translate=[],
    temperature=1.0,
    repeat_penalty=1.1,
    translate_repeat_penalty=1.1,
    chat_format="",
    n_ctx_default=8192,
    patches=["jinja2_loopcontrols"],
    use_chat_api=True,
)

PROFILES = {
    "aya": AYA,
    "gemma": GEMMA,
}


def get_profile(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise ValueError(
            f"Unknown model profile: {name}. Available: {list(PROFILES.keys())}"
        )
    return PROFILES[name]
