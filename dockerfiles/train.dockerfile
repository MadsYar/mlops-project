# Base image
FROM nvcr.io/nvidia/pytorch:24.12-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ADD . /app
WORKDIR /app
RUN uv sync --frozen
COPY uv.lock uv.lock
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ src/
COPY models models/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# # RUN uv pip install -r requirements.txt --no-cache-dir --verbose
# RUN pip install -r requirements.txt
# RUN pip install . --no-deps --no-cache-dir --verbose

# RUN dvc init --no-scm
# COPY .dvc/config .dvc/config
# COPY *.dvc .dvc/
# RUN dvc config core.no_scm true
# RUN dvc pull

ENTRYPOINT ["python", "-u", "src/danish_to_english_llm/train.py"]
