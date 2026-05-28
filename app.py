import os
import re
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ========== DeepSeek 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

KNOWLEDGE_FILE = "Gourmet_Knowledge_Base.txt"

def load_knowledge_text():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "暂无知识库。"

# ---------- 解析器（完全保留你的代码） ----------
def parse_data():
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    province_city_map = {}
    city_food_map = {}
    current_province = None
    current_city = None
    in_hot_section = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line == "直辖市" or line.startswith("直辖市"):
            in_hot_section = True
            continue

        if re.match(r'^[━─]+$', line):
            if in_hot_section:
                in_hot_section = False
            continue
        if line.startswith("格式：") or line.startswith("说明：") or line.startswith("中国城市美食知识库"):
            continue

        province_match = re.match(r'^【(.+?)】$', line)
        if province_match:
            province_name = province_match.group(1).strip()
            if province_name in province_city_map:
                current_province = province_name
                continue
            current_province = province_name
            if current_province not in province_city_map:
                province_city_map[current_province] = []
            current_city = None
            continue

        city_match = re.match(r'^([^：:]+)[：:]$', line)
        if city_match and current_province:
            city_name = city_match.group(1).strip()
            if city_name not in province_city_map[current_province]:
                province_city_map[current_province].append(city_name)
            current_city = city_name
            if current_city not in city_food_map:
                city_food_map[current_city] = []
            continue

        food_match = re.match(r'^\d+\.', line)
        if food_match and current_city is not None:
            content = line[line.find('.')+1:].strip()
            parts = [p.strip() for p in content.split('|')]
            if len(parts) >= 1:
                name = parts[0] if len(parts) > 0 else "未知"
                spicy = parts[1] if len(parts) > 1 else "未知"
                price = parts[2] if len(parts) > 2 else "未知"
                crowd = parts[3] if len(parts) > 3 else "未知"
                intro = parts[4] if len(parts) > 4 else ""
                city_food_map[current_city].append({
                    "name": name,
                    "spicy": spicy,
                    "price": price,
                    "crowd": crowd,
                    "intro": intro
                })

    valid_provinces = [p for p in province_city_map if province_city_map[p]]

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

# ---------- 路由（完全保留） ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/api/provinces")
def get_provinces():
    provinces, _, _ = parse_data()
    return jsonify({"provinces": provinces})

@app.route("/api/cities")
def get_cities():
    province = request.args.get("province")
    _, pc_map, _ = parse_data()
    cities = pc_map.get(province, [])
    return jsonify({"cities": cities})

@app.route("/api/foods")
def get_foods():
    city = request.args.get("city")
    if not city:
        return jsonify({"foods": []})
    _, _, cf_map = parse_data()
    foods = cf_map.get(city, [])
    return jsonify({"foods": foods})

@app.route("/api/province_foods")
def get_province_foods():
    province = request.args.get("province")
    if not province:
        return jsonify({"foods": []})
    _, pc_map, cf_map = parse_data()
    cities = pc_map.get(province, [])
    all_foods = []
    for city in cities:
        foods = cf_map.get(city, [])
        for f in foods:
            f_with_city = f.copy()
            f_with_city['city'] = city
            all_foods.append(f_with_city)
    return jsonify({"foods": all_foods})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "请问你想了解什么美食？"})

    knowledge = load_knowledge_text()
    if len(knowledge) > 80000:
        knowledge = knowledge[:80000] + "...(内容过长已截断)"

    system_prompt = f"""你是一个专业的美食推荐助手。你的回答必须基于以下提供的中国城市美食知识库。
如果用户的问题在知识库中没有明确答案，请诚实地说"根据现有知识库无法回答该问题"，不要编造信息。
请用换行符和列表组织回答，清晰易读。

知识库内容：
{knowledge}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
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
        print("API调用失败:", e)
        return jsonify({"answer": "抱歉，AI服务暂时不可用，请稍后再试。"})

# ========== 适配 Render 启动（唯一修改点） ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
