import importlib

modules = importlib.import_module("models.modules")
auto_encoder = importlib.import_module("models.auto_encoder")
lstm_classifier = importlib.import_module("models.classifier")
