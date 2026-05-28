import os
import re
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ========== DeepSeek 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-5679e07c8dba435e9f3661cf39939206")
BASE_URL = "https://api.deepseek.com/v1"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

KNOWLEDGE_FILE = "Gourmet_Knowledge_Base.txt"

def load_knowledge_text():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "暂无知识库。"

# ---------- 全新健壮解析器 ----------
def parse_data():
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    province_city_map = {}
    city_food_map = {}
    current_province = None
    current_city = None
    in_hot_section = False   # 标记是否在“直辖市”板块

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # 检测到“直辖市”标题，进入特殊模式，但直辖市省份会正常处理
        if line == "直辖市" or line.startswith("直辖市"):
            in_hot_section = True
            continue

        # 跳过各种分隔线
        if re.match(r'^[━─]+$', line):
            # 连续分隔线表示离开直辖市板块
            if in_hot_section:
                in_hot_section = False
            continue
        if line.startswith("格式：") or line.startswith("说明：") or line.startswith("中国城市美食知识库"):
            continue

        # 省份标题：【xxx】
        province_match = re.match(r'^【(.+?)】$', line)
        if province_match:
            province_name = province_match.group(1).strip()
            # 去重：如果当前省份已被记录，跳过（避免重复）
            if province_name in province_city_map:
                current_province = province_name
                continue
            current_province = province_name
            if current_province not in province_city_map:
                province_city_map[current_province] = []
            current_city = None
            print(f"[解析] 发现省份: {current_province}")
            continue

        # 城市标题：xxx： 或 xxx: （兼容中英文冒号）
        city_match = re.match(r'^([^：:]+)[：:]$', line)
        if city_match and current_province:
            city_name = city_match.group(1).strip()
            # 避免重复添加城市
            if city_name not in province_city_map[current_province]:
                province_city_map[current_province].append(city_name)
            current_city = city_name
            if current_city not in city_food_map:
                city_food_map[current_city] = []
            print(f"[解析] 在 {current_province} 下添加城市: {current_city}")
            continue

        # 美食条目：数字. 内容（使用宽松管道符分割）
        food_match = re.match(r'^\d+\.', line)
        if food_match and current_city is not None:
            # 去掉开头的数字和点
            content = line[line.find('.')+1:].strip()
            # 按 | 分割
            parts = [p.strip() for p in content.split('|')]
            # 确保至少有一个字段
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

    # 打印调试信息（关键！）
    print("\n=== 最终解析结果 ===")
    print("省份列表:", list(province_city_map.keys()))
    print("城市美食数量:")
    for city, foods in city_food_map.items():
        print(f"  {city}: {len(foods)} 道美食")
    print("==================\n")

    # 过滤掉没有城市的省份
    valid_provinces = [p for p in province_city_map if province_city_map[p]]

    # 省份固定顺序（去重后排序）
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

# ---------- 其余路由保持不变 ----------
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
    print(f"[API] 请求城市: {city}, 返回美食数量: {len(foods)}")
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

    system_prompt = f"""你是一个专业的美食推荐助手。你的回答必须基于以下提供的中国城市美食知识库。如果用户的问题在知识库中没有明确答案，请诚实地说"根据现有知识库无法回答该问题"，不要编造信息。你可以结合知识库中的菜名、辣度、人均价格、适用人群和简介来给出推荐。

请使用换行符和列表（如“• ”或“1. ”）组织回答，让内容清晰易读。

知识库内容：
{knowledge}

请用中文回答。"""
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",  # 或 deepseek-v4-flash
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

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"错误：找不到知识库文件 {KNOWLEDGE_FILE}")
    else:
        app.run(debug=True)
