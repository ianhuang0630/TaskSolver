
## Setup & Installation
```bash
# clone this repository
git clone 
pip install -e .
```

## (Optional) Ollama
Should you choose to swap out GPT-4V for the open-sourced models supported by Ollama, you should follow the download instructions [here](https://ollama.com/download).

If you're launching this on a (headless) server, you may have to start the server in the background:
```bash
ollama serve
```

Before using a model (e.g. `mistral`), make sure to pull it first:
```bash
ollama pull mistral
```


