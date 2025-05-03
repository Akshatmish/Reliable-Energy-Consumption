#!/bin/bash
gunicorn -w 2 -k gthread app:app
