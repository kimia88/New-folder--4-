import pyodbc
import logging
from typing import List, Dict, Any, Optional

class SqlServerDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn: Optional[pyodbc.Connection] = None
        self.cursor: Optional[pyodbc.Cursor] = None

    def connect(self) -> None:
        try:
            self.conn = pyodbc.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            logging.info("✅ اتصال به دیتابیس برقرار شد.")
        except pyodbc.Error as e:
            logging.error(f"❌ خطا در اتصال به دیتابیس: {e}")
            raise

    def disconnect(self) -> None:
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logging.info("🔌 ارتباط با دیتابیس قطع شد.")
        except pyodbc.Error as e:
            logging.error(f"❌ خطا در قطع ارتباط با دیتابیس: {e}")

    def test_table_exists(self, table_name: str) -> bool:
        if not self.cursor:
            logging.error("❌ اتصال به دیتابیس برقرار نیست.")
            return False
        try:
            query = """
            SELECT CASE WHEN EXISTS (
                SELECT * FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = ?
            ) THEN 1 ELSE 0 END
            """
            self.cursor.execute(query, (table_name,))
            exists = self.cursor.fetchone()[0]
            return exists == 1
        except pyodbc.Error as e:
            logging.error(f"❌ خطا در بررسی وجود جدول '{table_name}': {e}")
            return False

    def get_contents_for_seo(self) -> List[Dict[str, Any]]:
        if not self.cursor:
            logging.error("❌ اتصال به دیتابیس برقرار نیست.")
            return []
        try:
            query = """
                SELECT Id, Title
                FROM TblPureContent 
                WHERE LEN(Title) > 0
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            contents = [{"content_id": row.Id, "title": row.Title} for row in rows]
            return contents
        except pyodbc.Error as e:
            logging.error(f"❌ خطا در واکشی محتوا: {e}")
            return []

    def update_pure_content(self, content_id: int, new_title: str) -> bool:
        if not self.cursor or not self.conn:
            logging.error("❌ اتصال به دیتابیس برقرار نیست.")
            return False
        try:
            query = "UPDATE TblPureContent SET Title = ? WHERE Id = ?"
            self.cursor.execute(query, (new_title, content_id))
            self.conn.commit()
            logging.info(f"✅ عنوان ID {content_id} با موفقیت به‌روزرسانی شد.")
            return True
        except pyodbc.Error as e:
            logging.error(f"❌ خطا در به‌روزرسانی عنوان ID {content_id}: {e}")
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
