FROM python:3.10.12

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.5.1 \
  POETRY_HOME="opt/poetry"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# System deps:
RUN pip install "poetry==$POETRY_VERSION"
RUN mkdir /code
COPY document_processing/ /code/document_processing
WORKDIR /code/document_processing

#COPY document_processing/poetry.lock document_processing/pyproject.toml /code/ 

RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-root
RUN pip install "transformers[torch]"
RUN python -m nltk.downloader punkt

RUN echo "Acquire { http::User-Agent \"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0\";};" > /etc/apt/apt.conf
    
RUN apt-get update \
    && apt-get install -y \
        hdf5-tools \
        curl

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y update
RUN apt-get install git-lfs

WORKDIR /code/nlp_models
RUN git clone https://huggingface.co/gsarti/biobert-nli
RUN cd biobert-nli && git init && git lfs install --local && rm -rf .git

WORKDIR /code/nlp_models
RUN git clone https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
RUN cd multi-qa-MiniLM-L6-cos-v1 && git init && git lfs install --local && rm -rf .git

WORKDIR /code/document_processing

EXPOSE 8888
EXPOSE 9999

ENTRYPOINT ["poetry", "run", "run_process"]
