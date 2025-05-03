import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 2  # Reduced for Render free tier
threads = 2
timeout = 300  # 5 minutes to handle model loading
loglevel = "info"
accesslog = "-"
errorlog = "-"
worker_class = "gthread"  # WSGI worker for Flask
max_requests = 1000
max_requests_jitter = 50
keepalive = 5
