import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 4
threads = 2
timeout = 120
loglevel = "info"
accesslog = "-"
errorlog = "-"
worker_class = "uvicorn.workers.UvicornWorker"
max_requests = 1000
max_requests_jitter = 50
keepalive = 5
