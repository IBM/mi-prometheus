#!/usr/bin/env bash

cd docs

# create html pages
sphinx-build -b html source build
make html

# open web browser(s) to master table of content
firefox build/index.html
