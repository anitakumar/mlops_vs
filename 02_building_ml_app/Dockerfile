FROM python:3.12
WORKDIR /usr/local/app
RUN mkdir -p /usr/local/app/model

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY app.py ./app.py
COPY model/house_price_model.pkl ./model/house_price_model.pkl
EXPOSE 5000

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD ["python", "app.py"]