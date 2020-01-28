#!/bin/bash
docker run --rm -i -p 8082:8082 --network toxic_net --name model_container -v /path/to/data:/toxic_comments/data toxic_classifier:latest predict