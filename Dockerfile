FROM ubuntu:latest
      
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ADD . /toxic_comments
WORKDIR /toxic_comments

RUN pip install -r requirements.txt
