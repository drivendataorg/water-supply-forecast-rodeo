FROM --platform=linux/amd64 mambaorg/micromamba:1.5.3-bookworm-slim

USER root

ARG CPU_OR_GPU=gpu

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

COPY runtime/apt.txt apt.txt
RUN apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && xargs -a apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /apt.txt

COPY --chown=$MAMBA_USER:$MAMBA_USER runtime/conda-lock-${CPU_OR_GPU}.yml /tmp/conda-lock.yml
RUN micromamba install --name base --yes --file /tmp/conda-lock.yml && \
    micromamba clean --all --force-pkgs-dirs --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN mkdir -p /code_execution
RUN chown -R ${MAMBA_USER}:${MAMBA_USER} /code_execution

COPY data_reading /data_reading
RUN pip install /data_reading --no-cache-dir --no-deps

COPY runtime/entrypoint.sh /entrypoint.sh
COPY runtime/supervisor.py /code_execution/supervisor.py

WORKDIR /code_execution
USER ${MAMBA_USER}
COPY runtime/tests /code_execution/tests

CMD ["bash", "/entrypoint.sh"]
