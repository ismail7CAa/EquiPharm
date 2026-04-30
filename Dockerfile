ARG EQUIPHARM_PLATFORM=linux/amd64
FROM --platform=${EQUIPHARM_PLATFORM} mambaorg/micromamba:1.5.10

WORKDIR /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

ENV PYTHONPATH=/workspace \
    MPLBACKEND=Agg

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

CMD ["bash"]
