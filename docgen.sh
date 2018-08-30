#/bin/bash
cd docs
#sphinx-apidoc -f -e -o source ../models/
#sphinx-apidoc -f -e -o source ../problems/
#sphinx-apidoc -f -e -o source ../misc/

sphinx-build -b html source build
make html
firefox build/index.html
