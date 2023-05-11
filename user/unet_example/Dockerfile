FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY ./ /opt/app/

RUN python -m piptools sync requirements.txt


COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
