FROM ubuntu:latest

EXPOSE 8082
      
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git\
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


ADD . /toxic_classifier
ADD . /luigi_files/log_dir/
WORKDIR /toxic_classifier
RUN chmod +x train
RUN chmod +x predict
ENV PATH="/toxic_classifier:${PATH}"
ENV MLFLOW_TRACKING_URI http://mlflow_container:5000
ENV AWS_ACCESS_KEY_ID some_acsess_key_id
ENV AWS_SECRET_ACCESS_KEY some_secret_access_key

RUN pip install .
#RUN pip install -r requirements.txt

ENV PYTHONPATH="/toxic_classifier"
ENV PYTHONPATH="$PYTHONPATH:/toxic_classifier/src"
ENV LUIGI_CONFIG_PATH='/toxic_classifier/luigi.cfg'
ENV LANG C.UTF-8 






