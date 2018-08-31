#/bin/bash
# this scripts takes .rst files and builds html documentation pages.
# Fancy: open the web browser to the index page

cd docs
sphinx-build -b html source build
make html
firefox build/index.html
chrome build/index.html
