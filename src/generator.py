import jinja2
from llama_cpp import Llama

from src.profiles import ModelProfile


def _apply_patches(patches: list[str]) -> None:
    """Apply model-specific patches."""
    if "jinja2_loopcontrols" in patches:
        _original_env_init = jinja2.Environment.__init__

        def _patched_env_init(self, *args, **kwargs):
            extensions = set(kwargs.get("extensions", []))
            extensions.add("jinja2.ext.loopcontrols")
            kwargs["extensions"] = list(extensions)
            _original_env_init(self, *args, **kwargs)

        jinja2.Environment.__init__ = _patched_env_init


def build_prompt(profile: ModelProfile, chunks: list[str], question: str) -> str:
    context = "\n\n".join(chunks) if chunks else "(no context available)"
    return profile.rag_template.format(context=context, question=question)


class Generator:
    def __init__(
        self,
        model_path: str,
        profile: ModelProfile,
        n_ctx: int = 0,
        n_threads: int = 4,
    ):
        self._profile = profile
        _apply_patches(profile.patches)
        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx or profile.n_ctx_default,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        if profile.chat_format:
            kwargs["chat_format"] = profile.chat_format
        self._llm = Llama(**kwargs)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip Gemma 4 thinking channel tokens from output."""
        import re
        # If there's a response channel after thinking, extract it
        response_match = re.search(r"<\|channel>response\n(.*?)(?:<channel\|>|$)", text, flags=re.DOTALL)
        if response_match:
            return response_match.group(1).strip()
        # Remove complete thinking blocks
        cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        # Remove incomplete thinking (no closing tag — model ran out of tokens mid-thought)
        cleaned = re.sub(r"<\|channel>thought\n.*", "", cleaned, flags=re.DOTALL)
        # Remove any remaining channel tags
        cleaned = re.sub(r"<\|channel>[^<]*", "", cleaned)
        cleaned = re.sub(r"<channel\|>", "", cleaned)
        cleaned = cleaned.strip()
        # If stripping removed everything, the answer is in the thinking — extract it
        if not cleaned and text:
            # Remove just the channel markers and return the content
            fallback = re.sub(r"<\|channel>thought\n?", "", text)
            fallback = re.sub(r"<channel\|>", "", fallback)
            return fallback.strip()
        return cleaned

    def _complete(self, prompt: str, max_tokens: int, temperature: float,
                  repeat_penalty: float, stop: list[str]) -> str:
        """Route to chat API or completion API based on profile."""
        if self._profile.use_chat_api:
            messages = [{"role": "user", "content": prompt}]
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=repeat_penalty,
            )
            text = result["choices"][0]["message"]["content"].strip()
            cleaned = self._strip_thinking(text)
            return cleaned if cleaned else text  # fallback to raw if stripping removes everything
        else:
            result = self._llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=repeat_penalty,
                stop=stop,
            )
            return result["choices"][0]["text"].strip()

    def generate(
        self,
        chunks: list[str],
        question: str,
        max_tokens: int = 512,
    ) -> str:
        prompt = build_prompt(self._profile, chunks, question)
        return self._complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self._profile.temperature,
            repeat_penalty=self._profile.repeat_penalty,
            stop=self._profile.stop_rag,
        )

    def chat(self, message: str, system: str = "", max_tokens: int = 200) -> str:
        prompt = self._profile.chat_template.format(
            system=system, message=message
        )
        return self._complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self._profile.temperature,
            repeat_penalty=self._profile.repeat_penalty,
            stop=self._profile.stop_chat,
        )

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_tokens: int = 200
    ) -> str:
        prompt = self._profile.translate_template.format(
            source=source_lang, target=target_lang, text=text
        )
        stop = list(self._profile.stop_translate)
        if not stop:
            stop = [f"\n{source_lang}:", "\n\n", f"\n{target_lang}:"]
        return self._complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            repeat_penalty=self._profile.translate_repeat_penalty,
            stop=stop,
        )
