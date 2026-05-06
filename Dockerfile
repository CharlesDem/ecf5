FROM python:3.11-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-compile -r requirements-api.txt \
    && find "$VIRTUAL_ENV" -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find "$VIRTUAL_ENV" -type d \( -name "test" -o -name "tests" \) -prune -exec rm -rf {} + \
    && find "$VIRTUAL_ENV" -type f -name "*.pyc" -delete \
    && find "$VIRTUAL_ENV" -type f \( -name "*.pyi" -o -name "py.typed" \) -delete \
    && find "$VIRTUAL_ENV" -type f \( -name "RECORD" -o -name "INSTALLER" -o -name "REQUESTED" \) -delete \
    && find "$VIRTUAL_ENV" -type f -name "*.so" -exec strip --strip-unneeded {} + || true \
    && rm -rf "$VIRTUAL_ENV/lib/python3.11/site-packages/pip"* \
    && rm -rf "$VIRTUAL_ENV/lib/python3.11/site-packages/setuptools"*


FROM python:3.11-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/api

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /opt/venv /opt/venv
COPY api/ /app/api/

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]