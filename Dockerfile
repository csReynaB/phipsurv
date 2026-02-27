FROM mambaorg/micromamba:1.5.10

WORKDIR /app

COPY ML_env.yml /tmp/ML_env.yml
RUN micromamba create -y -n ml -f /tmp/ML_env.yml -c conda-forge \
    && micromamba clean --all --yes

# Copy as the micromamba user (prevents permission issues)
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src

# Safety: remove any egg-info that may have slipped in
RUN rm -rf src/*.egg-info

RUN micromamba run -n ml python -m pip install -U pip setuptools wheel \
    && micromamba run -n ml python -m pip install . --no-deps

#CMD ["micromamba", "run", "-n", "ml", "python", "-c", "import survival; print('ok')"]
#CMD ["micromamba", "run", "-n", "ml", "survival-trainTest", "--help"]
ENTRYPOINT ["micromamba", "run", "-n", "ml"]
CMD ["train_test", "--help"]
