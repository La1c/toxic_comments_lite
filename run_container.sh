#!/bin/bash
docker run -d -i -p 8082:8082 -v /path/to/toxic_data:/toxic_comments/data -e AWS_ACCESS_KEY_ID=ABC -e AWS_SECRET_ACCESS_KEY=DEF --network=host toxic_classifier:latest train