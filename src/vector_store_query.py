from langchain_chroma import Chroma
from langdetect import detect, DetectorFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader_pdf_embedding import *
import os
from .utils import get_resource_path

CHROMA_DB_DIR = get_resource_path("./multi_lang_chroma_db")  # Chroma向量库存储路径
CHUNK_SIZE = 512  # 文本分块大小
CHUNK_OVERLAP = 64  # 分块重叠长度
DetectorFactory.seed = 0  # 固定语言检测种子，结果稳定


def is_file_in_chroma_db(db, file_path):
    """
    检查指定文件是否已存在于Chroma向量库中
    参数：
        db: Chroma数据库实例
        file_path: 待检查的文件路径（如 "./论文1.pdf"）
    返回：
        bool: True（已存在）/False（不存在）
    """
    # 提取文件名（和原函数中source元数据保持完全一致）
    file_name = os.path.basename(file_path)

    try:
        # 核心：通过元数据过滤查询该文件的所有记录
        # where参数实现精准匹配source字段（存储的是文件名）
        query_results = db.get(
            where={"source": file_name}  # 匹配原函数添加的source元数据
        )

        # 如果查询结果中有id，说明该文件已存在
        has_records = len(query_results["ids"]) > 0
        if has_records:
            print(f"ℹ️ 文件 {file_name} 已存在于向量库中（共 {len(query_results['ids'])} 个文本块）")
        return has_records

    except Exception as e:
        print(f"⚠️ 检查文件 {file_name} 是否存在时出错：{str(e)}")
        return False


def build_multi_lang_chroma_db(doc_paths):
    """
    批量处理多语言论文（新增重复检查逻辑）：
    1. 逐个检测论文语言 → 对应模型编码
    2. 为文档添加语言元数据（lang: zh/en）
    3. 合并所有向量到同一个Chroma库
    4. 前置检查：跳过已存入的文件
    """
    # 初始化空的Chroma库（统一存储）
    db = Chroma(
        embedding_function=get_bge_embeddings("zh"),
        persist_directory=CHROMA_DB_DIR
    )
    try:
        for file_path in doc_paths:
            # ===== 新增：前置检查文件是否已存在 =====
            if is_file_in_chroma_db(db, file_path):
                print(f"⏭️ 跳过已存在的文件：{os.path.basename(file_path)}")
                continue
            # ======================================

            if not os.path.exists(file_path):
                print(f"❌ 文件不存在：{file_path}")
                continue

            # 步骤1：检测当前论文语言
            lang = detect_document_language(file_path)
            if lang == "unknown":
                print(f"⚠️ 无法检测{file_path}语言，使用跨语言模型")

            # 步骤2：加载对应模型
            embeddings = get_bge_embeddings(lang)

            # 步骤3：论文加载+过滤+分块（学术PDF优化）
            loader = PyMuPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            # 过滤无效文本（页眉页脚、乱码）
            filtered_docs = []
            for doc in docs:
                content = doc.page_content.strip()
                if len(content) > 20 and "��" not in content:
                    filtered_docs.append(doc)
            # 学术分块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", "$", "##", ",", ".", ]
            )
            split_docs = text_splitter.split_documents(filtered_docs)

            # 步骤4：添加元数据（语言+文件路径），关键！用于检索过滤
            for doc in split_docs:
                doc.metadata["lang"] = lang  # 语言元数据
                doc.metadata["source"] = os.path.basename(file_path)  # 来源论文名称

            # 步骤5：将当前论文的向量添加到统一Chroma库
            db.add_documents(documents=split_docs, embedding=embeddings)
            print(f"✅ 成功添加论文：{os.path.basename(file_path)} | 语言：{lang} | 文本块数：{len(split_docs)}")

        print(f"\n🎉 所有论文处理完成！向量库存储路径：{CHROMA_DB_DIR}")
        return db
    except Exception as e:
        print("输入pdf或txt格式有误，请检查pdf是否属于扫描图片")


# 多语言RAG检索函数（核心：查询语言匹配+元数据过滤）
def multi_lang_rag_search(query, db):
    """
    多语言检索逻辑：
    1. 检测查询语言 → 用对应模型生成查询向量
    2. 过滤同语言的论文片段 → 精准检索
    3. 支持跨论文联合检索
    """
    try:
        # 步骤1：检测查询语言
        query_lang = detect_text_language(query)
        print(f"🔍 检测到查询语言：{query_lang}")

        # 步骤2：获取对应模型，生成查询向量
        embeddings = get_bge_embeddings(query_lang)

        # 步骤3：构建带语言过滤的检索器
        retriever = db.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"lang": query_lang}  # 只检索同语言片段
            },
            embedding=embeddings
        )

        # 步骤4：执行检索（核心修复：替换旧方法）
        # 适配LangChain v0.1+ 新版接口
        relevant_docs = retriever.invoke(query)

        # 空结果处理
        if not relevant_docs:
            return f"❌ 未检索到{query_lang}语言的相关内容"

        # 步骤5：结构化拼接结果
        result = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", "未知论文")
            result.append(f"【相关片段{i + 1} | 来源：{source}】\n{doc.page_content}")
        return "\n\n".join(result)

    # 兜底：适配极旧版本LangChain（兼容get_relevant_documents）
    except AttributeError as e:
        if "invoke" in str(e):
            try:
                relevant_docs = retriever.get_relevant_documents(query)
                if not relevant_docs:
                    return f"❌ 未检索到{query_lang}语言的相关内容"
                result = []
                for i, doc in enumerate(relevant_docs):
                    source = doc.metadata.get("source", "未知论文")
                    result.append(f"【相关片段{i + 1} | 来源：{source}】\n{doc.page_content}")
                return "\n\n".join(result)
            except Exception as e2:
                return f"❌ 检索方法适配失败：{str(e2)}"
        return f"❌ 属性错误：{str(e)}"

    # 捕获其他异常（模型加载、向量库连接等）
    except Exception as e:
        return f"❌ 检索出错：{str(e)}"


