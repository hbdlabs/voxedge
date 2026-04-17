from fastembed import TextEmbedding

from src.profiles import ModelProfile


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str | None = None,
        cuda: bool = False,
    ):
        kwargs: dict = {"model_name": model_name, "cache_dir": cache_dir}
        if cuda:
            kwargs["cuda"] = True
        self._model = TextEmbedding(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into 384-dim vectors."""
        return [vec.tolist() for vec in self._model.embed(texts)]

    def active_providers(self) -> list[str]:
        """Return the ONNX Runtime providers actually in use by this session.

        Surfaces whether CUDA activated at runtime or silently fell back
        to CPU (which can happen if cuDNN is missing).
        """
        sess = getattr(self._model, "model", None)
        if sess is None or not hasattr(sess, "get_providers"):
            return []
        return sess.get_providers()


_SUPPORTED_DEVICES = {"cpu", "cuda"}


def build(
    profile: ModelProfile,
    model_name: str,
    cache_dir: str | None = None,
) -> Embedder:
    """Construct an Embedder according to the profile's embedder_device."""
    if profile.embedder_device not in _SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported embedder_device '{profile.embedder_device}' "
            f"for profile '{profile.name}'. Supported: {sorted(_SUPPORTED_DEVICES)}"
        )
    return Embedder(
        model_name=model_name,
        cache_dir=cache_dir,
        cuda=(profile.embedder_device == "cuda"),
    )
