# 基于本地RAG的学术代理
### 该项目是基于LangChain 1.2版本开发的学术智能代理，主要旨在解决大语言模型中的幻觉和数据隐私问题。支持TXT、PDF文档本地上传，构建RAG知识库，并设计Agent进行论文分析、比对和检索。可以通过Streamlit生成本地Web界面，进行可视化操作。该模型可以独立定制，以满足额外的需求，详细的项目结构参见README.md文件。项目下载后无法通过直接使用，需要您根据README配置文件。

### 对于使用LangChain及相关工具具有扎实的座席编程基础的用户，该项目甚至可以定制并重新用作基于座席的客户服务系统或其他座席应用程序。
## 使用示例
如图所示，这是最终运行的可视化操作界面，可以通过**Streamlit**提供的本地端口访问。您可以使用该代理搜索和下载多篇论文，分析不同论文的核心点，还可以分析您上传的本地知识库的内容。
![图片.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698305ccaf380f3b56b27fb8)
下载后的论文将放在项目的.arxiv_downloaded_pa​​pers文件夹中。
## 项目如何使用？
### 首先，您需要下载嵌入模型。由于体积较大（两个模型总共占用约5GB），因此无法上传到GitHub。
下载存储库后，需要分别修改两部分代码文件。该项目是 Naive RAG 类型的基于矢量的 RAG。本项目使用两种不同的嵌入模型进行内容索引，即中文模型BAAI/bge-large-zh-v1.5和英文模型BAAI/bge-large-en-v1.5，均由BAAI发布。您需要自行下载这些模型并将其放在相应的embedding_model文件夹中。

### 如下图所示
下面第一图展示了完整的项目结构图。
![图片.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-69830d50df7d4d1e4cae26fd)

![图片.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-69830d5eaf380f3b56b280a7)
这里推荐使用git命令和huggingface命令下载。在项目根目录下输入cmd，打开命令提示符。

```bash
git clone https://huggingface.co/BAAI/bge-large-zh ./bge-large-zh
git clone https://huggingface.co/BAAI/bge-large-en ./bge-large-en
```
### 配置.env文件中的内容

![Uploading image.png…]()

本项目使用DeepSeek V3.2作为基线版本。您可以直接从 DeepSeek 获取自己的 API 密钥并开箱即用。此外，您还可以使用各自的 API 密钥集成支持 OpenAI API 的其他大型模型。如果需要切换到不同推理模式的模型，则需要自行修改main.py中的结构体输出部分。

### 安装依赖项
建议您通过Conda创建新的虚拟环境，然后安装requirements.txt文件中指定的依赖项。
首先，通过Conda创建一个虚拟环境，然后激活它。
```bash
# 基本语法
conda create -n 环境名 python=版本号

# 示例：创建名为 bge_env 的 Python 3.9 环境
conda create -n bge_env python=3.9

# 激活环境
conda activate bge_env

# 验证激活
conda info --envs

# 进入项目目录，然后安装
pip install -r requirements.txt

# 如果 pip 安装慢，可以指定清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 完成所有准备步骤后，导航到项目根目录，打开命令提示符，然后输入相应的命令。然后就可以通过Streamlit启动该项目的可视化操作界面了（需要激活该项目的虚拟环境）

```bash
streamlit run main.py
```

![图片.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698318759c0db14c9eace2b3)

### 随后，您将进入项目的可视化操作界面。

![图片.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698318b8df7d4d1e4cae2802)




