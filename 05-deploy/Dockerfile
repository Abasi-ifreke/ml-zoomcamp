FROM python:3.12.5-slim-bookworm

# RUN pip install uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY .python-version pyproject.toml uv.lock /app/

RUN uv sync --locked

COPY predict.py model_C=1.0.bin /app/

EXPOSE 9696

# ENTRYPOINT [ "uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]

ENTRYPOINT [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]