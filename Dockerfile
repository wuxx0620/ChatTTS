FROM harbor.sensoro.com/pytorch/torchserve:0.11.0-gpu

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]