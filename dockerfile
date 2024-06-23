FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your application code
COPY requirements.txt .

# Install dependencies using pip within venv
RUN pip install -U -r requirements.txt

COPY . .

# Install uvicorn (assuming it's not in requirements.txt)
RUN pip install uvicorn

# Expose port (optional, adjust if needed)
EXPOSE 8000

# Command to run your application (adjust if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
