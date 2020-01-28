#!/bin/bash

line="*/3 * * * * run_container_pred.sh"
(crontab -l; echo "$line" ) | crontab -