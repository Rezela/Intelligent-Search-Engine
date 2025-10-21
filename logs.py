import logging
import datetime
import os
import uuid

def init_logger(log_name: str = "rag_engine") -> str:
    """
    初始化日志系统
    :param log_name: 日志文件前缀
    :return: 日志文件完整路径
    """
    # 确保 logs 文件夹存在
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    log_filename = os.path.join(
        log_dir,
        f"{log_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # 配置日志写入文件
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            # logging.StreamHandler()
        ]
    )

    logging.info("Logs saved to %s", log_filename)
    return log_filename



def new_query_id() -> str:
    """
    生成唯一查询 ID
    :return: UUID 字符串
    """
    QueryID = uuid.uuid4()
    logging.info(f"Query ID: %s", QueryID)
    return str(QueryID)