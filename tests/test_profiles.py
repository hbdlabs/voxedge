from src.profiles import ModelProfile, get_profile, PROFILES


def test_aya_profile_exists():
    profile = get_profile("aya")
    assert profile.name == "aya"
    assert "{context}" in profile.rag_template
    assert "{question}" in profile.rag_template
    assert "{message}" in profile.chat_template
    assert "{text}" in profile.translate_template
    assert len(profile.stop_rag) > 0
    assert "jinja2_loopcontrols" in profile.patches


def test_gemma_profile_exists():
    profile = get_profile("gemma")
    assert profile.name == "gemma"
    assert "{context}" in profile.rag_template
    assert "{question}" in profile.rag_template
    assert "{message}" in profile.chat_template
    assert "{text}" in profile.translate_template
    assert len(profile.stop_rag) > 0
    assert profile.use_chat_api is True
    assert profile.n_ctx_default == 8192


def test_unknown_profile_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown model profile"):
        get_profile("nonexistent")


def test_profiles_are_frozen():
    profile = get_profile("aya")
    import pytest
    with pytest.raises(AttributeError):
        profile.name = "changed"
