from langchain_chroma import Chroma
import os
import shutil
import time
import psutil  # 需安装：pip install psutil

CHROMA_DB_DIR = "./multi_lang_chroma_db"  # 你的向量库路径

# ===================== 方案1：极简版清空库内数据（跳过模型加载） =====================
def clear_chroma_db_fast():

    try:
        # 关键优化：用「空embedding」初始化（仅为适配接口，不加载模型）
        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.0]*1024]*len(texts)
            def embed_query(self, text):
                return [0.0]*1024

        # 初始化Chroma（无模型加载，1秒内完成）
        db = Chroma(
            embedding_function=DummyEmbeddings(),  # 虚拟embedding，跳过模型加载
            persist_directory=CHROMA_DB_DIR
        )

        # 步骤1：获取所有文档ID（分批处理）
        all_docs = db.get()
        all_doc_ids = all_docs["ids"]
        if not all_doc_ids:
            print("ℹ️ 向量库已为空，无需清空")
            return
        print(f"🔍 检测到 {len(all_doc_ids)} 个文本块，开始分批删除...")

        # 步骤2：分批删除（每批100个，避免锁等待）
        batch_size = 100
        for i in range(0, len(all_doc_ids), batch_size):
            batch_ids = all_doc_ids[i:i+batch_size]
            db.delete(ids=batch_ids)
            print(f"✅ 已删除第 {i//batch_size + 1} 批，共删除 {len(batch_ids)} 个文本块")
            time.sleep(0.1)  # 释放锁，避免sqlite3阻塞

        # 验证清空结果
        after_docs = db.get()
        print(f"\n🎉 清空完成！剩余文本块数：{len(after_docs['ids'])}")

    except Exception as e:
        print(f"❌ 清空失败：{str(e)}")
        # 兜底：直接调用方案2删除目录
        print("🔧 尝试强制删除整个向量库...")
        delete_chroma_db_force()

# ===================== 方案2：强制删除向量库（释放句柄+管理员权限） =====================
def release_file_handles():
    """
    释放Windows下Chroma的sqlite3文件句柄（关键！解决隐性占用）
    """
    try:
        # 遍历所有Python进程，关闭chroma.sqlite3的句柄
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                for file in proc.open_files():
                    if 'chroma.sqlite3' in file.path and CHROMA_DB_DIR in file.path:
                        print(f"🔓 释放文件句柄：{file.path}（进程PID：{proc.pid}）")
                        proc.kill()  # 关闭占用句柄的Python进程
                        time.sleep(1)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except Exception as e:
        print(f"⚠️ 释放句柄时警告：{str(e)}")

def delete_chroma_db_force():
    """
    强制删除向量库目录（先释放句柄，再删除）
    """
    try:
        # 步骤1：释放文件句柄
        release_file_handles()
        time.sleep(2)  # 等待句柄释放

        # 步骤2：强制删除目录
        if os.path.exists(CHROMA_DB_DIR):
            # 先清空目录内文件（避免权限不足）
            for root, dirs, files in os.walk(CHROMA_DB_DIR, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.chmod(file_path, 0o777)  # 赋予所有权限
                    os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    os.rmdir(dir_path)
            # 删除主目录
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            print(f"✅ 强制删除成功！已删除目录：{CHROMA_DB_DIR}")
        else:
            print(f"ℹ️ 向量库目录不存在：{CHROMA_DB_DIR}")

    except PermissionError:
        print("❌ 权限不足！请按以下步骤操作：")
        print("1. 以管理员身份运行Python/CMD；")
        print("2. 关闭所有打开的文件管理器窗口（尤其是向量库目录）；")
        print("3. 重新运行本代码。")
    except Exception as e:
        print(f"❌ 强制删除失败：{str(e)}")
