# 1. Start with a basic Python environment
FROM --platform=linux/amd64 python:3.10-slim

# 2. Set the main folder inside the box to /app
WORKDIR /app

# 3. Copy just the shopping list into the box
COPY requirements.txt .

# 4. Install all the libraries from the shopping list
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and models into the box
COPY src/ /app/src/
COPY models/ /app/models/

# 6. Set the final command to run automatically when the box is opened
CMD ["python", "src/run_inference.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]