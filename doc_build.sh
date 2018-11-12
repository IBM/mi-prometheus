#!/usr/bin/env bash

cd docs
rm -rf build

# create html pages
sphinx-build -b html source build
make html

# open web browser(s) to master table of content
if which firefox
then
    firefox build/index.html
fi
