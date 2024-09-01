from setuptools import setup, find_packages

setup(
    name="tasksolver",
    version="0.0.1",
    packages=find_packages(exclude=["scripts"]),
    install_requires=[
        "loguru",
	    "pillow",
        "bson",
        "requests",
        "openai",
        "streamlit",
        "flask",
        "flask_cors",
        "ollama",
        "anthropic",
        "google-generativeai"
    ]
)
