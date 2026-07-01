FROM mambaorg/micromamba:1.5.10

USER root

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/envs/equipharm/bin:/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace \
    MPLBACKEND=Agg

WORKDIR /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba create -y -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && mkdir -p /workspace \
    && chown -R $MAMBA_USER:$MAMBA_USER /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

USER $MAMBA_USER

CMD ["bash"]
