#!/usr/bin/env bash

# Define the log file
LOGFILE="build_and_run.txt"

# Clean and build, and log the output
make clean build > "$LOGFILE" 2>&1

# Run the program and log the output, appending to the existing log file
make run ARGS="-input=/home/coder/project/gaussianBlurFilterNPP/PGM_Images -output=/home/coder/project/gaussianBlurFilterNPP/PGM_Images/processed" >> "$LOGFILE" 2>&1
