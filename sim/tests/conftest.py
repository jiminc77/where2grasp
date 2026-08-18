def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: test that builds a real Genesis scene (needs the GPU / CUDA backend).",
    )
