FROM python:3.11-slim

# set working directory
WORKDIR /app

# install dependencies first (layer caching — reinstall only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# environment variables — real values injected at runtime, never hardcoded
ENV FLASK_ENV=production
ENV PORT=8000

# expose the port Flask runs on
EXPOSE 8000

# run the app
CMD ["python", "app.py"]