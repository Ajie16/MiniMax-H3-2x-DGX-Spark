ARG BASE_IMAGE=minimax-h3-dgx-spark:sm121-fp8
FROM ${BASE_IMAGE}

RUN python -m pip install --no-cache-dir "ray[default]==2.56.1"

COPY h3_multinode /opt/h3-multinode/h3_multinode
COPY patches/enable-ray-diffusion-executor.patch /tmp/enable-ray-diffusion-executor.patch
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p1 < /tmp/enable-ray-diffusion-executor.patch && \
    rm /tmp/enable-ray-diffusion-executor.patch
ENV PYTHONPATH=/opt/h3-multinode
