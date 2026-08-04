ARG BASE_IMAGE=minimax-h3-dgx-spark:sm121-fp8
FROM ${BASE_IMAGE}

ARG H3_UPSTREAM_BASE_IMAGE=vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e
ARG H3_COMPANION_REPO_COMMIT=8bd7628dbdb51a0ea00c301ddcb1a098874870e4
ARG H3_ACCEPTED_BASE_IMAGE_ID=sha256:2383642e221530d3dc26a8f8632c37e00470b051979f0845c2ec0ff9513e04b2
LABEL org.opencontainers.image.base.name="${H3_UPSTREAM_BASE_IMAGE}" \
      org.opencontainers.image.base.digest="sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e" \
      io.github.joeynyc.minimax-h3.companion-commit="${H3_COMPANION_REPO_COMMIT}" \
      io.github.joeynyc.minimax-h3.accepted-local-base-id="${H3_ACCEPTED_BASE_IMAGE_ID}"

RUN python -m pip install --no-cache-dir "ray[default]==2.56.1"

COPY h3_multinode /opt/h3-multinode/h3_multinode
COPY patches/enable-ray-diffusion-executor.patch /tmp/enable-ray-diffusion-executor.patch
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p1 < /tmp/enable-ray-diffusion-executor.patch && \
    rm /tmp/enable-ray-diffusion-executor.patch
ENV PYTHONPATH=/opt/h3-multinode
