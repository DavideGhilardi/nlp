# NLP project

Before running the notebook with the analysis, the following steps have to be executed:

1. Install required packages with `pip install -r requirements.txt``
2. Authenticate on Huggingface with `huggingface-cli login --token [TOKEN]`
3. Download Llama 2 7B from Huggingface with `huggingface-cli download meta-llama/Llama-2-7b-hf``

Beware that the model occupies **~30GB** on disk and requires at least **16GB** of RAM (GPU is recommended) for loading (in bf16).