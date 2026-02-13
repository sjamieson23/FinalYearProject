#!/bin/bash
nohup python3 -u Selecting.py > masterLogs.log 2>&1
nohup python3 -u SelectingLr.py > masterLogsLR.log 2>&1
