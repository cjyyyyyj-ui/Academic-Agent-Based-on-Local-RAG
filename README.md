# Academic Agent Based on Local RAG
### This project is an academic intelligent agent developed based on LangChain Version 1.2, primarily designed to address the issues of hallucinations and data privacy in large language models. It supports the local upload of TXT and PDF documents to build a RAG knowledge base, and the agent is engineered for paper analysis, comparison and retrieval. A local web interface can be generated via Streamlit for visual operation. The model can be customized independently to meet additional requirements, and the detailed project structure is available in the README.md file.Mainly for introductory basic learning.This project is not ready for direct download and use. Please configure it in accordance with the steps outlined in the README.

### For users with a solid foundation in agent programming using LangChain and related tools, this project can even be customized and repurposed as an agent-based customer service system or other agent applications.
## Usage Examples
As shown in the figure, this is the visual operation interface for the final run, which is available via a local port provided by **Streamlit**.You can use this agent to search and download multiple papers, analyze the core points across different papers, and also analyze the content of the local knowledge base you have uploaded.
![image.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698305ccaf380f3b56b27fb8)
Papers after being downloaded will be placed in the .arxiv_downloaded_papers folder of the project.
## How to use the project ？
### First, you need to download the embedding models. Due to their large size (the two models take up about 5 GB in total), they cannot be uploaded to GitHub.
After downloading the repository, you need to modify two parts of the code files separately. This project is a vector-based RAG of the Naive RAG type. Two different embedding models are used for content indexing in this project, namely the Chinese model BAAI/bge-large-zh-v1.5 and the English model BAAI/bge-large-en-v1.5, both released by BAAI. You need to download these models by yourself and place them in the corresponding embedding_model folder.

### As shown in the figure below
The first figure below shows the complete project structure diagram.
![image.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-69830d50df7d4d1e4cae26fd)

![image.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-69830d5eaf380f3b56b280a7)
Git commands and huggingface commands are recommended for downloading here.Enter cmd in the project root directory to open the command prompt.

```bash
git clone https://huggingface.co/BAAI/bge-large-zh ./bge-large-zh
git clone https://huggingface.co/BAAI/bge-large-en ./bge-large-en
```
### Configure the content in the .env file

<img width="945" height="278" alt="image" src="https://github.com/user-attachments/assets/24bc888b-96a9-46fc-abe9-aa19e242da58" />

This project uses DeepSeek V3.2 as the baseline version. You can obtain your own API key directly from DeepSeek and use it out of the box. Additionally, you can also integrate other large models that support the OpenAI API by using their respective API keys. If you need to switch to a model with a different reasoning mode, you will need to modify the structure output part in main.py by yourself.

### Install the dependencies
It is recommended that you create a new virtual environment via Conda and then install the dependencies specified in the requirements.txt file.
First, create a virtual environment via Conda and then activate it.
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
### Once all preparation steps are completed, navigate to the project root directory, open the command prompt, and enter the corresponding command. You can then launch the project's visual operation interface via Streamlit.（It is necessary to activate the virtual environment for this project.）

```bash
streamlit run main.py
```

![image.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698318759c0db14c9eace2b3)

### Subsequently, you will be directed to the project's visual operation interface.

![image.png](https://tc-cdn.processon.com/po/658679616ff9af23035e9c84-698318b8df7d4d1e4cae2802)




