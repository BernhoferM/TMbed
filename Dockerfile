FROM python:3.9-slim AS builder

ENV TZ=Europe/London

WORKDIR /opt/tmbed
COPY . /opt/tmbed

RUN pip install --no-cache-dir numpy==2.0.2 \
 && pip install --no-cache-dir h5py==3.13.0 \
 && pip install --no-cache-dir sentencepiece==0.2.0 \
 && pip install --no-cache-dir tqdm==4.67.1 \
 && pip install --no-cache-dir transformers==4.51.3 \
 && pip install --no-cache-dir typer==0.15.3 \
 && pip install --no-cache-dir torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir .

FROM python:3.9-slim

ENV TZ=Europe/London

COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRYPOINT ["tmbed"]
