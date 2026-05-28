# 先配置国内镜像，解决huggingface下载超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 所需库全部导入
import re
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# 初始化Flask网站应用
app = Flask(__name__)

# ---------------------- AI大模型配置 ----------------------
# DeepSeek密钥和接口地址
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
# 校验环境变量，未配置则直接报错终止运行
if not DEEPSEEK_API_KEY:
    raise ValueError("错误：未配置环境变量 DEEPSEEK_API_KEY，请先配置后再运行")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# 美食知识库文本文件路径
KNOWLEDGE_FILE = "Gourmet_Knowledge_Base.txt"

# ---------------------- RAG向量库初始化 ----------------------
# 加载语义向量模型
rag_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# 连接本地持久化向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# 获取建好的美食向量集合
chroma_collection = chroma_client.get_collection("food_knowledge")


# 读取本地知识库文本内容
def load_knowledge_text():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "暂无知识库。"


# 核心：解析知识库，把 省份-城市-美食 拆成映射关系
def parse_data():
    # 读取整个知识库所有行
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 存：省份对应哪些城市
    province_city_map = {}
    # 存：每个城市对应哪些美食
    city_food_map = {}
    # 临时记录当前读到的省份、城市
    current_province = None
    current_city = None
    # 标记是不是在读直辖市那一块
    in_hot_section = False

    # 逐行遍历解析
    for raw_line in lines:
        line = raw_line.strip()
        # 空行直接跳过
        if not line:
            continue

        # 识别到直辖市标题，标记一下
        if line == "直辖市" or line.startswith("直辖市"):
            in_hot_section = True
            continue

        # 跳过分隔线、说明文字这些没用的行
        if re.match(r'^[━─]+$', line):
            if in_hot_section:
                in_hot_section = False
            continue
        if line.startswith("格式：") or line.startswith("说明：") or line.startswith("中国城市美食知识库"):
            continue

        # 匹配省份格式 【xxx】
        province_match = re.match(r'^【(.+?)】$', line)
        if province_match:
            province_name = province_match.group(1).strip()
            # 重复省份就跳过
            if province_name in province_city_map:
                current_province = province_name
                continue
            current_province = province_name
            if current_province not in province_city_map:
                province_city_map[current_province] = []
            current_city = None
            print(f"[解析] 发现省份: {current_province}")
            continue

        # 匹配城市格式  xxx：
        city_match = re.match(r'^([^：:]+)[：:]$', line)
        if city_match and current_province:
            city_name = city_match.group(1).strip()
            # 加入对应省份的城市列表
            if city_name not in province_city_map[current_province]:
                province_city_map[current_province].append(city_name)
            current_city = city_name
            if current_city not in city_food_map:
                city_food_map[current_city] = []
            print(f"[解析] 在 {current_province} 下添加城市: {current_city}")
            continue

        # 匹配美食条目 1.xxx 这种开头
        food_match = re.match(r'^\d+\.', line)
        if food_match and current_city is not None:
            # 把序号后面的内容切出来
            content = line[line.find('.')+1:].strip()
            # 按 | 分割各个字段
            parts = [p.strip() for p in content.split('|')]
            # 给每个字段赋默认值，防止缺东西报错
            if len(parts) >= 1:
                name = parts[0] if len(parts) > 0 else "未知"
                spicy = parts[1] if len(parts) > 1 else "未知"
                price = parts[2] if len(parts) > 2 else "未知"
                crowd = parts[3] if len(parts) > 3 else "未知"
                intro = parts[4] if len(parts) > 4 else ""
                # 存入当前城市的美食列表
                city_food_map[current_city].append({
                    "name": name,
                    "spicy": spicy,
                    "price": price,
                    "crowd": crowd,
                    "intro": intro
                })

    # 解析完打印一下统计信息，方便看有没有问题
    print("\n=== 最终解析结果 ===")
    print("省份列表:", list(province_city_map.keys()))
    print("城市美食数量:")
    for city, foods in city_food_map.items():
        print(f"  {city}: {len(foods)} 道美食")
    print("==================\n")

    # 过滤掉没城市的空省份
    valid_provinces = [p for p in province_city_map if province_city_map[p]]

    # 固定省份显示顺序
    province_order = [
        "北京市", "上海市", "天津市", "重庆市",
        "安徽省", "福建省", "甘肃省", "广东省", "广西壮族自治区", "贵州省",
        "海南省", "河北省", "河南省", "黑龙江省", "湖北省", "湖南省",
        "吉林省", "江苏省", "江西省", "辽宁省", "内蒙古自治区", "宁夏回族自治区",
        "青海省", "山东省", "山西省", "陕西省", "四川省", "台湾省", "西藏自治区",
        "新疆维吾尔自治区", "云南省", "浙江省", "港澳台地区"
    ]
    provinces = sorted(valid_provinces, key=lambda x: province_order.index(x) if x in province_order else 999)

    return provinces, province_city_map, city_food_map


# RAG检索：根据用户问题，从向量库找最相关的知识库内容
def retrieve_related_context(question, top_k=3):
    # 把用户问题转成向量
    query_embedding = rag_model.encode(question).tolist()
    # 去向量库查相似度最高的几条
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where_document = {"$contains": question}
    )
    docs = results["documents"][0]
    # 控制台打印，方便看检索到了啥
    print("=== 问题：", question)
    print("=== 检索到的块：", docs)
    # 拼接成文本传给AI
    return "\n\n".join(docs)


# 网站首页
@app.route("/")
def index():
    return render_template("index.html")


# 聊天页面路由
@app.route("/chat")
def chat_page():
    return render_template("chat.html")


# 接口：获取所有省份
@app.route("/api/provinces")
def get_provinces():
    provinces, _, _ = parse_data()
    return jsonify({"provinces": provinces})


# 接口：根据省份拿下属城市
@app.route("/api/cities")
def get_cities():
    province = request.args.get("province")
    _, pc_map, _ = parse_data()
    cities = pc_map.get(province, [])
    return jsonify({"cities": cities})


# 接口：根据城市拿对应美食
@app.route("/api/foods")
def get_foods():
    city = request.args.get("city")
    if not city:
        return jsonify({"foods": []})
    _, _, cf_map = parse_data()
    foods = cf_map.get(city, [])
    print(f"[API] 请求城市: {city}, 返回美食数量: {len(foods)}")
    return jsonify({"foods": foods})


# 接口：获取一个省份下所有城市的全部美食
@app.route("/api/province_foods")
def get_province_foods():
    province = request.args.get("province")
    if not province:
        return jsonify({"foods": []})
    _, pc_map, cf_map = parse_data()
    cities = pc_map.get(province, [])
    all_foods = []
    # 遍历该省所有城市，汇总美食
    for city in cities:
        foods = cf_map.get(city, [])
        for f in foods:
            f_with_city = f.copy()
            f_with_city['city'] = city
            all_foods.append(f_with_city)
    return jsonify({"foods": all_foods})


# 聊天接口：接收用户问题，走RAG+AI回答
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    # 空问题直接回复
    if not question:
        return jsonify({"answer": "请问你想了解什么美食？"})

    # RAG先检索相关知识库
    context = retrieve_related_context(question)

    # 设定AI角色和规则，强制只能用知识库内容回答
    system_prompt = f"""你是一个专业的美食推荐助手。你的回答必须基于以下提供的中国城市美食知识库。如果用户的问题在知识库中没有明确答案，请诚实地说"根据现有知识库无法回答该问题"，不要编造信息。你可以结合知识库中的菜名、辣度、人均价格、适用人群和简介来给出推荐。

    请使用换行符和列表（如“• ”或“1. ”）组织回答，让内容清晰易读。

    知识库内容：
    {context}

    请用中文回答。"""
    try:
        # 调用DeepSeek生成回答
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print("DeepSeek API调用失败:", e)
        return jsonify({"answer": "抱歉，AI服务暂时不可用，请稍后再试。"})
# 程序入口启动
if __name__ == "__main__":
    # 先检查知识库文件在不在
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"错误：找不到知识库文件 {KNOWLEDGE_FILE}")
    else:
        app.run(debug=True)