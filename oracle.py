import oracledb

# Initialize the Instant Client (adjust path if different)
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_11_2")

DB_USER = "penta_survey"
DB_PASS = "555"
DB_DSN  = "//localhost:1522/orcl"  # service name matches your listener

try:
    conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
    cur = conn.cursor()

    # Show total rows
    cur.execute("SELECT COUNT(*) FROM survey_data")
    print("Row count:", cur.fetchone()[0])

    # Show first 5 rows
    cur.execute("SELECT * FROM survey_data WHERE ROWNUM <= 5")
    cols = [d[0] for d in cur.description]
    for r in cur:
        print(dict(zip(cols, r)))

    cur.close()
    conn.close()
    print("Connection & query OK")

except Exception as e:
    print("ERROR:", e)
