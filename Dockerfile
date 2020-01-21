FROM ubuntu:latest

EXPOSE 8082
      
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git\
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN adduser --disable-password worker

WORKDIR /home/worker


ADD . /toxic_comments
ADD . /luigi_files/log_dir/
WORKDIR /toxic_comments
RUN chmod +x train
USER worker
ENV PATH="/home/worker/.local/bin:${PATH}"
ENV PATH="/toxic_comments:${PATH}"
ENV MLFLOW_TRACKING_URI http://mlflow_container:5000
ENV AWS_ACCESS_KEY_ID some_acsess_key_id
ENV AWS_SECRET_ACCESS_KEY some_secret_access_key
COPY --chown=worker:worker . .


RUN pip install --user -r requirements.txt

ENV PYTHONPATH="/toxic_comments/"
ENV PYTHONPATH="$PYTHONPATH:/toxic_comments/src"
ENV LUIGI_CONFIG_PATH='/toxic_comments/luigi.cfg'
ENV LANG C.UTF-8 






