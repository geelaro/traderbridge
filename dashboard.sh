#!/usr/bin/env bash
exec pipenv run streamlit run dashboard.py --server.port 8501 --server.headless true "$@"
