import jinja2
from llama_cpp import Llama

# Enable {% break %} / {% continue %} in Jinja2 templates embedded in GGUF models
jinja2.defaults.DEFAULT_NAMESPACE  # ensure module loaded
_original_env_init = jinja2.Environment.__init__


def _patched_env_init(self, *args, **kwargs):
    extensions = set(kwargs.get("extensions", []))
    extensions.add("jinja2.ext.loopcontrols")
    kwargs["extensions"] = list(extensions)
    _original_env_init(self, *args, **kwargs)


jinja2.Environment.__init__ = _patched_env_init


PROMPT_TEMPLATE = """You are a helpful assistant at a community knowledge kiosk.
Use ONLY the provided context to answer the question.
Answer in the same language as the question.
If the context does not contain enough information to answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(chunks: list[str], question: str) -> str:
    context = "\n\n".join(chunks) if chunks else "(no context available)"
    return PROMPT_TEMPLATE.format(context=context, question=question)


class Generator:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
    ):
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
            chat_format="raw",
        )

    def generate(
        self,
        chunks: list[str],
        question: str,
        max_tokens: int = 512,
    ) -> str:
        prompt = build_prompt(chunks, question)
        result = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.3,
            stop=["\nQuestion:", "\n\n\n", "\nNote:", "(Note:"],
        )
        return result["choices"][0]["text"].strip()
