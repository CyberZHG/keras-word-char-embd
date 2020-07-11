#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_wc_embd tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_wc_embd --with-doctest