from fastapi import FastAPI, Query, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
import oracledb
import csv
from io import StringIO

app = FastAPI()

# Oracle Client init
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_11_2")

DB_USER = "penta_survey"
DB_PASS = "555"
DB_DSN  = "//localhost:1522/orcl"

# API Key settings
API_KEY = "mysecret123"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dependency to check API Key (header or query param)
def get_api_key(api_key_header: str = Security(api_key_header), api_key: str = None):
    key = api_key_header or api_key
    if key == API_KEY:
        return key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# Root endpoint - Row count
@app.get("/")
def read_root(api_key: str = Security(get_api_key)):
    try:
        conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM survey_data")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"status": "success", "row_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Fetch limited data
@app.get("/data")
def get_data(limit: int = Query(5, ge=1, le=100), api_key: str = None, api_key_header: str = Security(api_key_header)):
    key = api_key_header or api_key
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM survey_data WHERE ROWNUM <= {limit}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"status": "success", "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Download all data as CSV
@app.get("/download")
def download_csv(api_key: str = None, api_key_header: str = Security(api_key_header)):
    key = api_key_header or api_key
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT * FROM survey_data")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(cols)
        writer.writerows(rows)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=survey_data.csv"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Search data by state and/or gender
@app.get("/search")
def search_data(state: str = None, gender: str = None, api_key: str = None, api_key_header: str = Security(api_key_header)):
    key = api_key_header or api_key
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cur = conn.cursor()
        query = "SELECT * FROM survey_data WHERE 1=1"
        params = {}
        if state:
            query += " AND state = :state"
            params["state"] = state
        if gender:
            query += " AND gender = :gender"
            params["gender"] = gender

        cur.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"status": "success", "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------- Prediction logic ----------------
def make_prediction(input_data):
    """
    Replace this function with real ML model prediction logic.
    For now, it just returns a dummy string based on input.
    """
    # Example: pretend we predict a category based on first column
    return f"Predicted value based on input: {input_data}"

@app.get("/predict")
def predict(api_key: str = None, api_key_header: str = Security(api_key_header)):
    key = api_key_header or api_key
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        # Fetch input data from DB (example: first row)
        conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT * FROM survey_data WHERE ROWNUM = 1")
        data = cur.fetchone()
        cur.close()
        conn.close()

        if not data:
            return {"status": "error", "message": "No data found for prediction"}

        # Prepare input (adjust according to your table structure)
        input_data = [data[1], data[2]]  # Example: using 2 columns

        # Call the prediction function
        prediction = make_prediction(input_data)

        return {"status": "success", "prediction": prediction}

    except Exception as e:
        return {"status": "error", "message": str(e)}
