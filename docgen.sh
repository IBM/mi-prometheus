#/bin/bash
# this scripts takes .rst files and builds html documentation pages.
# Also creates code coverage

cd docs
# create code coverage
sphinx-build -b coverage source build
cp build/python.txt source/python.rst

# create html pages
sphinx-build -b html source build
make html

# open web browser(s) to master table of content
firefox build/index.html
chrome build/index.html
