def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs to download a config/tokenizer from the Hugging Face Hub")
