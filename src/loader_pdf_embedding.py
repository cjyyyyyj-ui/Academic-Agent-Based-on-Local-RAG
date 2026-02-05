from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langdetect import detect, DetectorFactory
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from .utils import get_resource_path
import os
import re
from dotenv import load_dotenv

env_path = get_resource_path(".env")
load_dotenv(env_path)

# 固定检测种子，提升langdetect结果的一致性（可选）
DetectorFactory.seed = 0

# 缓存模型实例：避免重复加载zh/en模型（提升效率）
# 注：当前代码未使用模型缓存，保留注释便于后续扩展
model_cache = {}


def detect_text_language(text):
    """检测文本语言，优先取前1000字符"""
    # 1. 文本非空校验
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "unknown"

    # 2. 移除换行符和制表符，合并多个空格为一个
    text = text.replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r'\s+', ' ', text)

    # 3. 截取前1000字符，避免超长文本影响效率
    detect_text = text[:1000] if len(text) > 1000 else text

    try:
        # 4. 执行语言检测
        lang = detect(detect_text)

        # 5. 语言映射，仅保留zh/en/unknown
        lang_map = {
            'zh': 'zh',  # 中文
            'zh-cn': 'zh',
            'zh-tw': 'zh',
            'en': 'en',  # 英文
        }

        return lang_map.get(lang, "unknown")
    except Exception as e:
        print(f"语言检测失败: {e}")
        return "unknown"


def detect_document_language(file_path):
    """检测单篇论文的语言"""
    # 1. 先校验文件是否存在且为文件（非文件夹）
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print("错误：文件路径无效或文件不存在")
        return "unknown"

    text = ""
    try:
        # 2. 解析PDF文件
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])

        # 3. 解析TXT文件（支持utf-8和GBK编码，提升兼容性）
        elif file_path.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                # utf-8解析失败时，尝试GBK编码
                loader = TextLoader(file_path, encoding="gbk")
                docs = loader.load()
            text = docs[0].page_content

        # 4. 不支持的文件格式
        else:
            return "unknown"

    except Exception as e:
        print(f"文档解析失败: {e}")
        return "unknown"

    # 5. 调用文本语言检测
    return detect_text_language(text)

# 加载对应语言的BGE模型（缓存机制，避免重复加载）
def get_bge_embeddings(language):
    if language in model_cache:
        return model_cache[language]  # 直接返回缓存的模型

    z_model = get_resource_path("./embedding_model/bge-large-zh-v1.5")
    e_model = get_resource_path("./embedding_model/bge-large-en-v1.5")
    model_configs = {
        "zh": {
            "model_name": z_model,
            "query_instruction": "为这个句子生成表示以用于检索相关文章："
        },
        "en": {
            "model_name": e_model,
                "query_instruction": "Represent this sentence for searching relevant passages: "
        },
        "unknown": {
            "model_name": z_model,
            "query_instruction": "为这个句子生成表示以用于检索相关文章："
        }
    }
    config = model_configs[language]
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config["model_name"],
        model_kwargs={"device": os.getenv("EMBUDDING_DEVICE")},  # 无GPU改cpu
        encode_kwargs={"normalize_embeddings": True},
        query_instruction=config["query_instruction"]
    )
    model_cache[language] = embeddings  # 缓存模型
    print(f"✅ 加载并缓存模型：{config['model_name']}")
    return embeddings




