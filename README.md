## To do
- [ ] Modify clean_text function in tagifai/data.py to return dataframe
- [ ] Avoid using snorkel. It has very small utility for our project, but huge size

### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```
### Commands to launch the application server
- Dev server
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app
``` 
- Production server
```bash
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

### Topics covered so far 
1. Testing code 
1. Testing data 
1. Testing model