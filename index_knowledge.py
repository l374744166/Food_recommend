# 设置 huggingface 国内镜像，解决网络下载失败问题
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
from sentence_transformers import SentenceTransformer
import chromadb

# 按城市对知识库进行分块，保证每个城市的美食信息完整不拆分
def split_by_city(text):
    chunks = []
    current_province = None
    current_city = None
    current_content = []

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 匹配省份格式：【省份名】
        prov_match = re.match(r'^【(.+?)】$', line)
        if prov_match:
            # 如果已有保存的城市数据，先存入分块列表
            if current_city and current_content:
                chunk = f"省份：{current_province}\n城市：{current_city}\n" + "\n".join(current_content)
                chunks.append(chunk)
                current_content = []

            current_province = prov_match.group(1)
            continue

        # 匹配城市格式：城市名：
        city_match = re.match(r'^([^：:]+)[：:]$', line)
        if city_match:
            # 保存上一个城市的内容
            if current_city and current_content:
                chunk = f"省份：{current_province}\n城市：{current_city}\n" + "\n".join(current_content)
                chunks.append(chunk)
                current_content = []

            current_city = city_match.group(1)
            continue

        # 匹配美食条目（以数字序号开头）
        if current_city and line.startswith(('1.', '2.', '3.', '4.', '5.')):
            current_content.append(line)

    # 处理最后一个城市的数据
    if current_city and current_content:
        chunk = f"省份：{current_province}\n城市：{current_city}\n" + "\n".join(current_content)
        chunks.append(chunk)

    return chunks

# 主程序执行
print("加载向量模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 读取美食知识库文件
with open("Gourmet_Knowledge_Base.txt", "r", encoding="utf-8") as f:
    full = f.read()

# 执行按城市分块
chunks = split_by_city(full)
print(f"分块完成，总计 {len(chunks)} 个城市数据块")

# 生成文本向量
embeddings = model.encode(chunks).tolist()
# 连接本地向量数据库
client = chromadb.PersistentClient(path="./chroma_db")

# 重建向量集合
try:
    client.delete_collection("food_knowledge")
except:
    pass

coll = client.create_collection("food_knowledge")
coll.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"c{i}" for i in range(len(chunks))]
)

print("向量数据库构建完成！")
# 打印前3个分块示例
for i, c in enumerate(chunks[:3]):
    print(f"\n分块 {i}:")
    print(c[:200] + "...")