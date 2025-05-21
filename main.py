import asyncio
import logging
import urllib3
from services.sql_server_database import SqlServerDatabase
from services.llm_service import QService
from services.seo_service import SEOServiceAdvanced

# غیرفعال کردن هشدارهای InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_database_connection():
    SERVER = "45.149.76.141"
    DATABASE = "ContentGenerator"
    USERNAME = "admin"
    PASSWORD = "Nightmare123!@#"

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD}"
    )
    db = SqlServerDatabase(connection_string)
    return db

def setup_services(db):
    SESSION_HASH = "amir"
    q_service = QService(session_hash=SESSION_HASH)
    seo_service = SEOServiceAdvanced(db=db, q_service=q_service)
    return seo_service

def test_table_existence(db):
    try:
        if not db.test_table_exists('TblPureContent'):
            logger.error("❌ جدول 'TblPureContent' پیدا نشد.")
            return False
        logger.info("✅ جدول 'TblPureContent' موجود است.")
        return True
    except Exception as e:
        logger.error(f"❌ خطا در چک کردن جدول: {e}")
        return False

async def main():
    db = setup_database_connection()
    try:
        logger.info("🔌 در حال اتصال به دیتابیس...")
        db.connect()
        logger.info("✅ اتصال برقرار شد.")

        if not test_table_existence(db):
            return

        seo_service = setup_services(db)

        logger.info("🚀 شروع فرآیند بهینه‌سازی عناوین برای سئو...")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, seo_service.optimize_titles)

    except Exception as e:
        logger.exception(f"❌ خطای کلی در اجرای برنامه: {e}")

    finally:
        db.disconnect()
        logger.info("🔌 ارتباط با دیتابیس قطع شد.")

if __name__ == "__main__":
    asyncio.run(main())
