FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 pytest==8.4.2 pytest-timeout==2.4.0 && \
    pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY worker.py artifacts.py /runner/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
USER 65534:65534
ENTRYPOINT ["python", "/runner/worker.py"]
