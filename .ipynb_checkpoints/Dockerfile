FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt

RUN pip install fastapi pydantic uvicorn scikit-learn pandas

COPY predict_v2.py .



EXPOSE 9696


CMD ["uvicorn", "predict_v2:app", "--host", "0.0.0.0", "--port", "9696"]