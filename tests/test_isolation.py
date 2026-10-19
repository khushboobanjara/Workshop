from app.config import settings


def test_tests_run_with_clean_defaults_regardless_of_your_env_file_or_shell():
    """If this fails, something in your .env or shell is leaking into the tests (see tests/conftest.py)."""
    assert settings.auth_enabled is False and settings.demo_mode is False and settings.env == "development"
    assert settings.api_key.get_secret_value() == "" and settings.bootstrap_admin_user == ""
    from sqlalchemy.engine import make_url
    url = make_url(settings.database_url)
    assert settings.db_user == "" and settings.db_password.get_secret_value() == ""          # app DB settings did not leak in
    assert url.get_backend_name() == "sqlite" or "test" in (url.database or "").lower()      # never a real database
