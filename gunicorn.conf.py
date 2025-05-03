import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 2
threads = 2
timeout = 300
loglevel = "info"
accesslog = "-"
errorlog = "-"
worker_class = "gthread"
max_requests = 1000
max_requests_jitter = 50
keepalive = 5
