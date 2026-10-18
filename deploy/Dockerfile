FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY openperfusion ./openperfusion
RUN pip install --no-cache-dir -e ".[dicom,register]" pynetdicom pillow
EXPOSE 11112
ENTRYPOINT []
CMD ["openperfusion", "serve", "--out", "/data"]
