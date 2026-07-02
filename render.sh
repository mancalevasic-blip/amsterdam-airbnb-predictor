#!/bin/bash
# Sets libomp path before Quarto launches Python so XGBoost can load.
export DYLD_LIBRARY_PATH="/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/sklearn/.dylibs"
quarto render ml-analysis.qmd "$@"
