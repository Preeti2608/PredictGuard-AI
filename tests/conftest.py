import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def app_instance(settings: Settings):
    app = create_app(settings)
    return app


@pytest.fixture(scope="session")
def client(app_instance):
    with TestClient(app_instance) as test_client:
        yield test_client
