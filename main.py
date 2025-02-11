import os
import configparser
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from openai import AzureOpenAI

# 從 config.ini 讀取配置
config = configparser.ConfigParser()
config.read("config.ini")

# 從 azure_openai 區塊讀取設定
ENDPOINT_URL = config.get("azure_openai", "ENDPOINT_URL", fallback="https://aidid-openai.openai.azure.com/")
DEPLOYMENT_NAME = config.get("azure_openai", "DEPLOYMENT_NAME", fallback="gpt-4o-mini")
SUBSCRIPTION_KEY = config.get("azure_openai", "AZURE_OPENAI_API_KEY", fallback="REPLACE_WITH_YOUR_KEY_VALUE_HERE")
SEARCH_ENDPOINT = config.get("azure_openai", "SEARCH_ENDPOINT", fallback="https://aidid-gpt.search.windows.net")
SEARCH_KEY = config.get("azure_openai", "SEARCH_KEY", fallback="put your Azure AI Search admin key here")
SEARCH_INDEX = config.get("azure_openai", "SEARCH_INDEX_NAME", fallback="aidid_houses")

# 從 api 區塊讀取 API token
API_TOKEN = config.get("api", "API_TOKEN", fallback="MY_SECRET_TOKEN")

# 初始化 Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=ENDPOINT_URL,
    api_key=SUBSCRIPTION_KEY,
    api_version="2024-05-01-preview"
)

# 全域對話歷史 (初始包含系統提示)
conversation_history = [
    {
        "role": "system",
        "content": (
            "【機器人使用說明】\n"
            "語言要求：請使用繁體中文回覆所有問題及回應。\n"
            "系統角色與任務：您的主要目標是根據客戶的需求與偏好，從房屋資料中找出最匹配的房屋選項並提供詳細資訊。\n"
            "請利用資料中的各項欄位（例如：價格、坪數、地點、房型等）來評估與推薦最適合客戶的房屋。\n"
            "請在回答時依下列格式回應：\n\n"
            "【推薦區域】：<區域名稱>\n"
            "【推薦公園】：<推薦公園及說明>\n"
            "【房型】：<房型及說明>\n"
            "【詳細資訊】：\n"
            "   名稱：<房屋名稱>\n"
            "   地址：<完整地址>\n"
            "   價格：<價格>\n"
            "   坪數：<建築坪數或室內面積>\n"
            "   格局：<房型或格局>\n"
            "   屋齡：<屋齡>\n"
            "   樓層：<樓層資訊>\n"
            "   社區：<社區或小區名稱>\n"
            "   特色：<其他特色說明>\n\n"
            "另外，請附上描述及思考過程，列出考慮的因素，例如坪數、房型、周邊環境及其他條件。"
        )
    }
]

# 定義 Pydantic 模型以驗證請求負載
class ChatRequest(BaseModel):
    message: str

# 初始化 FastAPI 應用程式
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, x_api_token: str = Header(...)):
    # 驗證 API Token
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")

    # 將使用者的新訊息加入對話歷史
    conversation_history.append({"role": "user", "content": request.message})

    # 呼叫 Azure OpenAI Chat API
    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=conversation_history,
            max_tokens=2000,
            temperature=0.68,
            top_p=0.71,
            frequency_penalty=0,
            presence_penalty=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from OpenAI: {e}")

    # 取得助手回覆 (使用 dot notation)
    reply = completion.choices[0].message.content

    # 將助手回覆加入對話歷史
    conversation_history.append({"role": "assistant", "content": reply})

    # 回傳助手回覆與完整對話歷史
    return {"reply": reply, "conversation": conversation_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
