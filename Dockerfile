ARG BASE_IMAGE=minimax-h3-dgx-spark:sm121-fp8
FROM ${BASE_IMAGE}

ARG H3_UPSTREAM_BASE_IMAGE=vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e
ARG H3_COMPANION_REPO_COMMIT=8bd7628dbdb51a0ea00c301ddcb1a098874870e4
ARG H3_ACCEPTED_BASE_IMAGE_ID=sha256:e498adce4d08a27d88f7c2a23a50563d508815cb78f12442a765326e661e2146
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=$PIP_INDEX_URL \
    PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST
LABEL org.opencontainers.image.base.name="${H3_UPSTREAM_BASE_IMAGE}" \
      org.opencontainers.image.base.digest="sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e" \
      io.github.joeynyc.minimax-h3.companion-commit="${H3_COMPANION_REPO_COMMIT}" \
      io.github.joeynyc.minimax-h3.accepted-local-base-id="${H3_ACCEPTED_BASE_IMAGE_ID}"

RUN python -m pip install --no-cache-dir "ray[default]==2.56.1"

# comfy-kitchen provides the INT8 ConvRot kernels used by the H3_QUANTIZATION=int8_convrot path.
RUN python -m pip install --no-cache-dir "comfy-kitchen==0.2.31"

COPY h3_multinode /opt/h3-multinode/h3_multinode
COPY patches/comfy_kitchen_int8.py /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/comfy_kitchen_int8.py
COPY patches/minimax_h3_transformer.py /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py
COPY patches/enable-ray-diffusion-executor.patch /tmp/enable-ray-diffusion-executor.patch
COPY patches/enable-lora-catalog.patch /tmp/enable-lora-catalog.patch
COPY patches/fix-h3-sigma-nfe.patch /tmp/fix-h3-sigma-nfe.patch
COPY patches/enable-lora-wrap.patch /tmp/enable-lora-wrap.patch
COPY patches/allow-file-audio-url.patch /tmp/allow-file-audio-url.patch
COPY patches/allow-mixed-ref-inputs.patch /tmp/allow-mixed-ref-inputs.patch
COPY patches/async-video-encode.patch /tmp/async-video-encode.patch
COPY patches/ref-encode-posterior-mean.patch /tmp/ref-encode-posterior-mean.patch
COPY patches/vision-tower-fp32.patch /tmp/vision-tower-fp32.patch
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p1 < /tmp/enable-ray-diffusion-executor.patch && \
    patch -p1 < /tmp/enable-lora-catalog.patch && \
    patch -p1 < /tmp/fix-h3-sigma-nfe.patch && \
    patch -p1 < /tmp/enable-lora-wrap.patch && \
    patch -p1 < /tmp/allow-file-audio-url.patch && \
    patch -p1 < /tmp/allow-mixed-ref-inputs.patch && \
    patch -p1 < /tmp/async-video-encode.patch && \
    patch -p1 < /tmp/ref-encode-posterior-mean.patch && \
    patch -p1 < /tmp/vision-tower-fp32.patch && \
    rm /tmp/enable-ray-diffusion-executor.patch /tmp/enable-lora-catalog.patch /tmp/fix-h3-sigma-nfe.patch /tmp/enable-lora-wrap.patch /tmp/allow-file-audio-url.patch /tmp/allow-mixed-ref-inputs.patch /tmp/async-video-encode.patch /tmp/ref-encode-posterior-mean.patch /tmp/vision-tower-fp32.patch
# Full-file overwrite: Ref2VA multi-reference pipeline (images + videos + audios,
# Comfy ordering). Applied after the diff patches; no overlap with them.
COPY patches/minimax_h3_pipeline.py /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py
ENV PYTHONPATH=/opt/h3-multinode
