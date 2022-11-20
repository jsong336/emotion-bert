FROM python:3.10.0

ARG UID=1000
ARG GID=1000

# group: mllab 
# user: emotion_bert
RUN groupadd -g ${GID} mllab && \
    useradd -m -g ${GID} -u ${UID} emotion_bert

USER ${UID}

RUN pip install --user pip && \
    pip install --user poetry==1.2.2

WORKDIR /emotion_bert 
ENV PATH="${PATH}:/home/emotion_bert/.local/bin"

COPY pyproject.toml . 
COPY poetry.lock .

RUN poetry install --no-interaction --no-dev --no-root

COPY notebook notebook 
COPY tests tests

# Jupyterlab port 
EXPOSE 8888
CMD [ "poetry", "run", "jupyter-lab", "--ip", "0.0.0.0", "--port", "8888" ] 