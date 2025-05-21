import pyodbc

class SqlServerDatabase:
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = pyodbc.connect(self.conn_str)
        return self.conn

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def test_table_exists(self, table_name):
        conn = self.connect()
        cursor = conn.cursor()
        query = "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
        cursor.execute(query, (table_name,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def select(self, query, params=None):
        conn = self.connect()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def execute(self, query, params=None):
        """
        برای کوئری‌هایی مثل UPDATE, INSERT, DELETE
        """
        conn = self.connect()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        cursor.close()

    def update_pure_content(self, content_id, new_title):
        query = "UPDATE dbo.TblPureContent SET Title = ? WHERE Id = ?"
        self.execute(query, (new_title, content_id))


# نحوه ساخت connection string برای SQL Server
def create_connection_string(server, database, username, password):
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )


# تست اتصال و اجرای یک select ساده
if __name__ == "__main__":
    SERVER = "45.149.76.141"
    DATABASE = "ContentGenerator"
    USERNAME = "admin"
    PASSWORD = "Nightmare123!@#"

    conn_str = create_connection_string(SERVER, DATABASE, USERNAME, PASSWORD)
    db = SqlServerDatabase(conn_str)

    try:
        # اتصال به دیتابیس
        db.connect()
        print("✅ اتصال به دیتابیس برقرار شد.")

        # بررسی وجود جدول
        if db.test_table_exists('TblPureContent'):
            print("✅ جدول TblPureContent موجود است.")
        else:
            print("❌ جدول TblPureContent پیدا نشد.")

        # اجرای یک کوئری SELECT نمونه
        rows = db.select("SELECT TOP 5 Id, Title FROM dbo.TblPureContent")
        for row in rows:
            # دسترسی با اندیس مطمئن‌تر است
            print(f"Id: {row[0]}, Title: {row[1]}")

        # نمونه آپدیت کردن یک رکورد (آیدی فرضی 1)
        db.update_pure_content(1, "عنوان جدید بهینه شده")
        print("✅ آپدیت عنوان با موفقیت انجام شد.")

    except Exception as e:
        print(f"❌ خطا در اتصال یا اجرای کوئری: {e}")

    finally:
        db.disconnect()
        print("🔌 اتصال به دیتابیس قطع شد.")
