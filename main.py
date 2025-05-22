import asyncio
import logging
import urllib3
from services.sql_server_database import SqlServerDatabase
from services.llm_service import QService
from services.seo_service import SEOServiceAdvanced

# غیرفعال کردن هشدارهای InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_database_connection() -> SqlServerDatabase:
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

def setup_services(db: SqlServerDatabase) -> SEOServiceAdvanced:
    SESSION_HASH = "amir"
    q_service = QService(session_hash=SESSION_HASH)
    seo_service = SEOServiceAdvanced(db=db, q_service=q_service)
    return seo_service

def test_table_existence(db: SqlServerDatabase) -> bool:
    try:
        if not db.test_table_exists('TblPureContent'):
            logger.error("❌ جدول 'TblPureContent' پیدا نشد.")
            return False
        logger.info("✅ جدول 'TblPureContent' موجود است.")
        return True
    except Exception as e:
        logger.exception(f"❌ خطا در بررسی وجود جدول: {e}")
        return False

async def main():
    db = setup_database_connection()
    connected = False

    try:
        logger.info("🔌 در حال اتصال به دیتابیس...")

        try:
            await asyncio.wait_for(asyncio.to_thread(db.connect), timeout=10)
            connected = True
            logger.info("✅ اتصال به دیتابیس برقرار شد.")
        except asyncio.TimeoutError:
            logger.error("❌ اتصال به دیتابیس تایم‌اوت شد.")
            return
        except Exception as e:
            logger.exception(f"❌ خطا در اتصال به دیتابیس: {e}")
            return

        if not test_table_existence(db):
            return

        seo_service = setup_services(db)

        logger.info("🚀 شروع فرآیند بهینه‌سازی عناوین برای سئو...")

        # گرفتن محتوا از دیتابیس
        contents = await asyncio.to_thread(db.get_contents_for_seo)
        if not contents:
            logger.warning("⚠️ داده‌ای برای بهینه‌سازی پیدا نشد.")
            return

        # فراخوانی بهینه‌سازی
        results = await asyncio.to_thread(seo_service.optimize_titles, contents)

        logger.info("✅ فرآیند بهینه‌سازی پایان یافت.")
        logger.info(f"📊 تعداد عناوین بهینه‌شده: {len(results)}")

        # ✅ نمایش نتایج بهینه‌سازی
        for res in results:
            logger.info(f"🎯 Content ID: {res['content_id']} | Optimized Title: {res['optimized_title']}")

    except Exception as e:
        logger.exception(f"❌ خطای کلی در اجرای برنامه: {e}")

    finally:
        if connected:
            try:
                db.disconnect()
                logger.info("🔌 ارتباط با دیتابیس قطع شد.")
            except Exception as e:
                logger.error(f"❌ خطا در قطع ارتباط با دیتابیس: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
