# Base image
FROM nvcr.io/nvidia/pytorch:24.12-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY uv.lock uv.lock

COPY src/ src/
COPY models models/
COPY configs/ configs/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

# RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r requirements_dev.txt
RUN pip install --no-cache-dir -e .

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true
RUN dvc pull


ENTRYPOINT ["python", "-u", "src/danish_to_english_llm/train.py"]
