FROM vllm/vllm-openai:latest

RUN pip install fastapi uvicorn boto3 httpx --no-cache-dir

COPY serve.py /opt/ml/code/serve.py

ENV PORT=8080
ENV MODEL_DIR=/opt/ml/model

EXPOSE 8080

ENTRYPOINT ["python3", "/opt/ml/code/serve.py"]