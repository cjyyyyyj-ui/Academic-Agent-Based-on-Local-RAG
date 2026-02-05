import streamlit as st
import os
import sys
import shutil
import  re
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langdetect import detect, DetectorFactory
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
import chromadb
import arxiv
import requests
from dotenv import load_dotenv

# 引入你的自定义模块
from src import (
    build_multi_lang_chroma_db,
    multi_lang_rag_search,
    detect_text_language,
    detect_document_language,
    get_bge_embeddings,
    clear_chroma_db_fast,
    get_resource_path
)

# --- 页面配置 ---
st.set_page_config(
    page_title="Multi-Lang Academic Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
#初始化数据库
doc_paths = []
db = build_multi_lang_chroma_db(doc_paths)

# 读取.env文件
env_path = get_resource_path(".env")
load_dotenv(env_path)


# ---------------------------------------------------------
@tool
def multi_lang_rag_search_tool(query: str) -> str:
    """
    多语言学术论文检索工具，支持中英文论文检索。
    功能：
    1. 自动检测用户查询的语言（中文/英文）；
    2. 返回检索到的内容及来源论文名称。的知识点、数据、结论等内容。
    3. 若用户指定查询某篇论文或论文之间进行对比，自动将用户的查询要求转化为相对应的语言，中文论文使用中文输入查询，英文论文使用英文输入查询
    """
    #适用场景：用户询问上传的中英文论文中
    try:
        return multi_lang_rag_search(query,db=db)
    except Exception as e:
        # 增加异常处理，避免工具调用崩溃
        return f"❌ 检索工具执行失败：{str(e)}"



@tool
def fetch_arxiv_pdf_download_tool(query: str, num: int, save_dir: str = "./arxiv_downloaded_papers") -> str:
    """
    根据用户要求搜索 ArXiv 上的前num的篇权威论文，并将它们的 PDF 下载到指定目录。
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=num,
        sort_by=arxiv.SortCriterion.Relevance
    )

    download_results = []
    os.makedirs(save_dir, exist_ok=True)

    print(f"Executing search for: {query}...")

    for idx, result in enumerate(client.results(search), 1):
        pdf_url = result.pdf_url
        title = result.title
        arxiv_id = result.entry_id.split("/")[-1]
        abstract = result.summary
        try:
            print(f"Downloading PDF {idx}: {title}...")
            valid_filename = f"{arxiv_id}_{title.replace('/', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')}.pdf"
            save_path = os.path.join(save_dir, valid_filename)

            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            success_msg = (
                f"--- Paper {idx} Download Success ---\n"
                f"标题: {title}\n"
                f"ArXiv ID: {arxiv_id}\n"
                f"保存路径: {os.path.abspath(save_path)}\n"
                f"摘要: {abstract}\n"
                f"--- End ---\n"
            )
            download_results.append(success_msg)

        except Exception as e:
            error_msg = f"--- Paper {idx} Download Failed ---\nTitle: {title}\nError: {str(e)}\n--- End ---\n"
            download_results.append(error_msg)

    return "\n\n".join(download_results)


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    if len(messages) <= 10:
        return None
    first_msg = messages[0]
    recent_messages = messages[-9:] if len(messages) % 2 == 0 else messages[-10:]
    new_messages = [first_msg] + recent_messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


# 初始化连接大模型
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=os.getenv("DEEPSEEK_TEMPERATURE"),
        max_tokens=os.getenv("DEEPSEEK_MAX_TOKENS")
    )


llm = get_llm()

# Agent Prompt
custom_prompt = """
    你是多语言学术论文分析智能体，用户已上传中英文论文。
    核心规则：
    1. 若用户要求分析论文内容则必须调用 multi_lang_rag_search_tool 工具检索论文内容为准，可结合自身知识输出，但严厉禁止凭空编造无依据的论文！
    2. 回答语言与用户查询语言一致（中文查询→中文回答，英文查询→英文回答）
    3. 回答时需标注内容来源的论文名称。
    4. 只有当用户提到要求搜索论文并下载时才调用fetch_arxiv_pdf_download_tool工具进行论文搜索并下！下载完毕后分析论文摘要。若用户使用中文查询论文自动将中文关键字转换为英文输入工具再查询。
    """
tools = [multi_lang_rag_search_tool, fetch_arxiv_pdf_download_tool]


# 初始化Agent
@st.cache_resource
def init_agent():
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=custom_prompt,
        middleware=[trim_messages],
        checkpointer=InMemorySaver()
    )


Academic_agent = init_agent()


# --- Streamlit UI 逻辑 ---

def main():
    # 标题栏
    st.title("🎓 多语言学术论文分析助手")
    st.caption("基于 RAG 与 ArXiv 的智能科研伙伴")

    # --- 侧边栏：知识库管理 ---
    with st.sidebar:
        st.header("📚 知识库管理")

        # 1. 数据库控制
        st.subheader("1. 数据库操作")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空表数据", help="保留结构，清空内容"):
                try:
                    clear_chroma_db_fast()
                    st.toast("✅ 表数据已清空", icon="🧹")
                except Exception as e:
                    st.error(f"操作失败: {e}")

        with col2:
            if st.button("删除数据库", help="完全删除数据库文件"):
                try:
                    if os.path.exists("chroma_db"):
                        shutil.rmtree("chroma_db")
                    # 清空 session
                    st.session_state.db_instance = None
                    st.toast("✅ 数据库已删除", icon="🗑️")
                    st.rerun()  # 强制刷新页面以更新状态
                except Exception as e:
                    st.error(f"删除失败: {e}")

        st.divider()

        # 2. 文件上传与构建
        st.subheader("2. 上传论文 (PDF/TXT)")
        uploaded_files = st.file_uploader(
            "拖拽文件到此处构建知识库",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )

        if st.button("🚀 构建/更新 向量库", type="primary"):
            if not uploaded_files:
                st.warning("请先上传文件！")
            else:
                with st.status("正在处理文档...", expanded=True) as status:
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)

                    doc_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        doc_paths.append(file_path)
                        st.write(f"已缓存: {uploaded_file.name}")

                    st.write("正在构建向量索引...")

                    # 构建数据库
                    new_db = build_multi_lang_chroma_db(doc_paths)

                    # 【核心】更新 Session State
                    st.session_state.db_instance = new_db

                    status.update(label="✅ 知识库构建完成！", state="complete", expanded=False)
                    st.toast("知识库已就绪，可以开始提问了！", icon="🎉")

        # 显示当前状态
        if st.session_state.get("db_instance") is not None:
            st.success("🟢 知识库状态：已加载")
        else:
            st.info("⚪ 知识库状态：未初始化")

    # --- 主聊天区域 ---

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 处理用户输入
    if prompt := st.chat_input("请输入你的研究问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            config: RunnableConfig = {"configurable": {"thread_id": "1"}}

            try:
                with st.spinner("Agent 正在思考与检索..."):
                    result = Academic_agent.invoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config=config
                    )

                if 'messages' in result and len(result['messages']) > 0:
                    final_message = result['messages'][-1]
                    if isinstance(final_message, AIMessage):
                        full_response = final_message.content
                    else:
                        full_response = str(final_message)

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Agent 运行出错: {str(e)}")


# --- 启动入口 ---
if __name__ == "__main__":
    main()