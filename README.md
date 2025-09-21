# LLM Transformer HuggingFace FastAPI

A FastAPI-based REST API for text generation using HuggingFace Transformers.  
This project enables you to interact with modern language models via an easy-to-use API, primed for both local development and free cloud deployment.

---

## 🚀 Purpose

- **Experiment** with LLM text generation using HuggingFace models.
- **Learn** about production-ready ML API deployment (CI/CD, cloud, best practices).
- **Free Hosting:** Built for hosting on platforms like [Render.com](https://render.com/) for zero-cost public demos.

---

## 🏗️ Architecture Overview

```
[ Client (curl, browser) ]
           |
           v
   [ FastAPI Server (Uvicorn) ]
           |
           v
[ HuggingFace Transformers Model ]
           |
           v
     [ Response JSON ]
```

- **FastAPI**: Handles HTTP requests, input validation, and docs.
- **HuggingFace Transformers**: Provides state-of-the-art language models for text generation.
- **Uvicorn**: ASGI server for fast async performance.
- **Render.com**: Free cloud hosting with auto-deploy from GitHub.

---

## ⚙️ Setup & Local Development

1. **Clone the repository**
    ```bash
    git clone https://github.com/Atifmahmood12/llm-transformer-huggingface.git
    cd llm-transformer-huggingface
    ```

2. **Create and activate virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the API locally**
    ```bash
    uvicorn api.main:app --reload
    ```

5. **Test the API**
    - Open docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    - Or use:
        ```bash
        curl -X POST "http://127.0.0.1:8000/generate" \
          -H "Content-Type: application/json" \
          -d '{"prompt": "Once upon a time,","max_new_tokens": 50,"temperature": 1.0,"do_sample": true}'
        ```

---

## ☁️ Free Deployment (Render.com)

1. **Push your code to GitHub**

2. **Sign up at [Render.com](https://render.com/)**

3. **Create a New Web Service**  
    - **Repo:** Connect your GitHub repo  
    - **Build Command:** `pip install -r requirements.txt`  
    - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port 10000`  
    - **Python Version:** 3.11  
    - **Port:** 10000

4. **Deploy!**  
    - Render will build and deploy your app.  
    - You’ll get a public HTTPS URL like: `https://llm-transformer-huggingface.onrender.com`

---

## 🔄 CI/CD with GitHub Actions

- Every push to `main` runs a CI workflow to check code and install dependencies.
- Render auto-deploys on every push to `main`.

Workflow file: `.github/workflows/ci.yml`

---

## 🧠 How model configuration affects behavior

- **prompt**: The input text you provide; sets the context for generation.
- **max_new_tokens**: Controls the length of the generated output (higher = more text).
- **temperature**: Increases randomness (higher = more creative/unpredictable output, lower = more deterministic).
- **do_sample**: Enables random sampling of tokens (set to `false` for greedy decoding, `true` for more creative outputs).
- **Model choice** (in code): Changing the HuggingFace model (e.g. `"gpt2"`) alters output style, quality, and capabilities.

**To change model:**  
Edit the model name in `api/main.py`:
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")  # Change "gpt2" to any HF model
```

---

## 📁 File Structure

```
llm-transformer-huggingface/
│
├── api/
│   ├── __init__.py   # Python package marker (must commit)
│   └── main.py       # FastAPI app and endpoints
│
├── requirements.txt  # Python dependencies
├── .github/workflows/ci.yml  # GitHub Actions workflow (CI)
├── README.md
└── ... (other files)
```

---

## 📝 Notes

- **`__init__.py`**: Required for Python to treat `api/` as a module. Always commit this file.
- **Security**: This is for learning—app is public if deployed!
- **Extending**: Add authentication, model upload, or more endpoints as you grow.

---

## 📬 Contact

- **Author:** [Atifmahmood12](https://github.com/Atifmahmood12)

---

## 🏁 Happy experimenting with LLMs and FastAPI!