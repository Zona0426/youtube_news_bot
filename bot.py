import html
import io
import logging
import os
import re
import tempfile
from dotenv import load_dotenv
from google import genai
from groq import Groq
import markdown
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

# ----------------------------------------------------
# 模塊 1：環境變數載入與身分驗證
# ----------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY or not GROQ_API_KEY:
    raise ValueError(
        "❌ 請確認 .env 中已填妥 TELEGRAM_BOT_TOKEN、GEMINI_API_KEY 與 GROQ_API_KEY"
    )

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 初始化 API 客戶端
ai_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def is_authorized(user_id: int | str) -> bool:
    if not ALLOWED_USER_ID:
        return True
    return str(user_id) == str(ALLOWED_USER_ID)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_authorized(user.id):
        await update.message.reply_text("⛔ 存取拒絕：你不是授權使用者。")
        return

    welcome_message = (
        f"👋 你好，{html.escape(user.first_name)}！\n\n"
        "我是你的 <b>YouTube 影片深度研報助理</b> 🤖\n\n"
        "📌 <b>功能亮點</b>：\n"
        "• 支援高精度字幕提取與 Groq Whisper 極速轉錄\n"
        "• 採用 Gemini 3.8-Flash 深度剖析因果邏輯與個股亮點\n"
        "• <b>手機最佳體驗</b>：自動輸出排版精美的 <code>.html</code> 研報，點開即讀無井字號干擾！\n\n"
        "👉 請傳送一個 YouTube 影片連結開始吧！"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")


# ----------------------------------------------------
# 模塊 2：YouTube 網址驗證與 Video ID 提取
# ----------------------------------------------------
def extract_youtube_video_id(text: str) -> str | None:
    patterns = [
        r"(?:v=|\/v\/|embed\/|shorts\/|live\/)([a-zA-Z0-9_-]{11})",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


# ----------------------------------------------------
# 模塊 3：提取內容（字幕優先，Groq Whisper 為備案）
# ----------------------------------------------------
def get_youtube_transcript(video_id: str) -> str | None:
    """嘗試取得字幕，相容新版物件屬性與舊版字典"""
    target_languages = ["zh-TW", "zh-Hant", "zh", "zh-CN", "zh-Hans", "en"]
    try:
        ytt_api = YouTubeTranscriptApi()
        data = ytt_api.fetch(video_id, languages=target_languages)
        texts = []
        for item in data:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(str(item))
        return " ".join(texts)
    except Exception as e:
        logger.info(f"無法直接抓取字幕 ({video_id}): {e}")
        return None


def download_audio_stream(youtube_url: str, output_dir: str) -> str:
    output_path = os.path.join(output_dir, "audio.m4a")
    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "quiet": True,
    "no_warnings": True,
    "cookiefile": "cookies.txt",
    "extractor_args": {"youtube": {"player_client": ["web"]}},
}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


def transcribe_audio_with_groq(audio_path: str) -> str:
    """透過 Groq Whisper-large-v3 進行高速轉錄"""
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"🎙 上傳至 Groq Whisper，檔案大小：{file_size_mb:.2f} MB")

    with open(audio_path, "rb") as file_obj:
        transcription = groq_client.audio.transcriptions.create(
            file=file_obj,
            model="whisper-large-v3",
            response_format="text",
            language="zh",
        )
    return str(transcription)


# ----------------------------------------------------
# 模塊 4：純文字深度研報分析（Gemini 3.8-Flash 通道）
# ----------------------------------------------------
PROMPT_ANALYSIS = """
你是一位頂級證券研究副總裁與個人知識庫專家。請以極度詳盡、嚴謹、具備深刻商業與投資洞察的角度，深度剖析這部影片的完整內容。

【撰寫原則】：
1. 拒絕空泛概括，嚴禁草率精簡！請以深度長文筆記的規格撰寫，全文請維持在 1,500 ~ 2,500 字左右的繁體中文深度報告。
2. 保留影片中提及的所有「具體數據、百分比、歷史對比、專有名詞、財報指標、因果推論脈絡與反方風險」。
3. 請使用標準 Markdown 語法排版，層次分明。

---

請依據以下結構進行完整剖析：

# 📌 影片主題深度精讀與商業筆記

## 🎯 一、 核心主題與一句話全局結論
- **核心主旨**：（精煉且精準描述整部影片欲探討的核心難題或趨勢）
- **關鍵結論**：（講者最後給出的具體定調或市場走向判斷）

## 🔍 二、 完整論證脈絡與章節深度剖析
（請根據影片進行結構化分章，每章節皆須交代「因果關係」與「關鍵數據支撐」）
- **【議題一：背景成因與當前市場狀態】**
  - 詳細交代當前市場現象與底層邏輯推動力。
  - 列出講者提及的具體佐證、圖表或歷史週期對照。
- **【議題二：核心轉折點與爭議焦點】**
  - 深度解析市場存在的不同分歧、主要矛盾點。
  - 講者針對此問題的推論過程與邏輯盲點檢視。
- **【議題三：中長期推演與趨勢發展】**
  - 未來 1~3 季或未來數年的產業/總經走向預測。
  - 關鍵的實體經濟或企業營運反饋指標。

## 💡 三、 投資人視角的關鍵洞察與風險警示
- **關鍵觀察指標（Checklist）**：（列出後續必須緊盯的 3~5 個具體指標或時間點）
- **下行風險提示（Downside Risk）**：（講者提及或隱含的潛在黑天鵝、邏輯失靈條件）
- **可落地之行動指南**：（針對一般投資人或研究員的具體策略建議）

---

# 🏢 影片提及公司與個股深度剖析表

若影片中提及任何企業、個股、ETF 或全球龍頭（包含口語簡稱、台股、美股），請整理成下方 Markdown 表格。若確實未提及任何公司，請填寫「本影片未涉及特定公司」。

| 公司名稱 | 股票代號 (台/美) | 核心產業類別 | 競爭優勢與核心業務 | 影片分析重點與潛在催化劑/風險 |
| :--- | :--- | :--- | :--- | :--- |
| (例) 台積電 | 2330.TW / TSM | 半導體製造 / 晶圓代工 | 先進製程市占超過 90%，CoWoS 封裝定價權高 | 剖析 2 奈米資本支出進度，影片認為毛利率有望維持 53% 以上，需關注地緣政治風險 |

請確保代號精準，業務描述具體扎實。
"""


def generate_notes_from_text(full_text: str) -> str:
    """調用付費版 Gemini 3.8-Flash 模型生成深度長篇研報"""
    MODEL_NAME = "gemini-3.8-flash"
    logger.info(
        f"🧠 正在調用 Gemini ({MODEL_NAME}) 處理文本（逐字稿共 {len(full_text):,} 字）..."
    )

    try:
        response = ai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                PROMPT_ANALYSIS,
                f"【以下為影片完整文字逐字稿】：\n{full_text}",
            ],
        )

        if not response.text or len(response.text.strip()) == 0:
            raise RuntimeError("Gemini 回傳內容為空")

        logger.info(
            f"✅ Gemini 研報產出成功！報告字數：{len(response.text):,} 字"
        )
        return response.text

    except Exception as e:
        logger.error(f"❌ Gemini 調用失敗: {e}")
        raise e


# ----------------------------------------------------
# 模塊 5：Markdown 轉精美自適應 HTML（手機直讀無井字號）
# ----------------------------------------------------
def convert_md_to_styled_html(md_content: str, title: str) -> str:
    """將 Markdown 轉為適合手機閱讀的精美自適應 HTML 網頁"""
    # 啟用表格與程式碼區塊擴充
    body_html = markdown.markdown(
        md_content, extensions=["tables", "fenced_code"]
    )

    html_template = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --border-color: #e2e8f0;
            --table-header: #f8fafc;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #0b0f19;
                --card-bg: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --primary: #38bdf8;
                --primary-light: #1e3a8a33;
                --border-color: #334155;
                --table-header: #0f172a;
            }}
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 12px;
            font-size: 16px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 24px 20px;
            border-radius: 18px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06);
        }}
        h1 {{
            font-size: 1.5rem;
            color: var(--primary);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 12px;
            margin-top: 0;
            line-height: 1.4;
        }}
        h2 {{
            font-size: 1.25rem;
            margin-top: 32px;
            margin-bottom: 16px;
            padding: 10px 14px;
            background: var(--primary-light);
            border-left: 5px solid var(--primary);
            border-radius: 6px;
            line-height: 1.4;
        }}
        h3 {{
            font-size: 1.1rem;
            margin-top: 20px;
            margin-bottom: 8px;
            color: var(--text-main);
        }}
        ul, ol {{
            padding-left: 20px;
            margin: 10px 0;
        }}
        li {{
            margin-bottom: 8px;
        }}
        strong {{
            color: var(--primary);
            font-weight: 700;
        }}
        p {{
            margin: 12px 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: var(--border-color);
            margin: 28px 0;
        }}
        /* 自適應股票剖析表格容器 */
        .table-wrapper {{
            width: 100%;
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            -webkit-overflow-scrolling: touch;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            text-align: left;
            min-width: 500px;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
        }}
        th:last-child, td:last-child {{
            border-right: none;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        th {{
            background-color: var(--table-header);
            font-weight: 600;
            color: var(--primary);
            white-space: nowrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        {body_html}
    </div>
</body>
</html>"""
    # 自動將 table 外面包一層可滑動的 div，確保手機端不破版
    html_template = html_template.replace(
        "<table>", '<div class="table-wrapper"><table>'
    ).replace("</table>", "</table></div>")
    return html_template


# ----------------------------------------------------
# 模塊 6：發送聊天室預覽與 HTML 研報附件
# ----------------------------------------------------
async def send_summary_and_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    markdown_text: str,
    video_id: str,
):
    """將報告轉成精美 HTML 檔案並發送附件給使用者"""
    # 產出 HTML 網頁
    styled_html = convert_md_to_styled_html(
        markdown_text, f"YouTube 深度研報 - {video_id}"
    )
    file_bytes = io.BytesIO(styled_html.encode("utf-8"))
    filename = f"YouTube_研報_{video_id}.html"
    file_bytes.name = filename

    preview_snippet = markdown_text[:350].strip()
    if len(markdown_text) > 350:
        preview_snippet += "\n\n...(點擊下方檔案即可在手機以精美網頁開啟完整長文與表格)"

    chat_id = update.effective_chat.id

    # 1. 聊天室文字預覽
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"📊 <b>研報摘要速覽</b>：\n\n<pre>{html.escape(preview_snippet)}</pre>",
        parse_mode="HTML",
    )

    # 2. 發送 .html 文件附件
    await context.bot.send_document(
        chat_id=chat_id,
        document=file_bytes,
        filename=filename,
        caption=(
            f"📱 <b>手機專屬精美研報</b>：<code>{filename}</code>\n"
            "<i>💡 點擊上方檔案，手機將自動以瀏覽器開啟漂亮排版（支援深色模式與自適應表格）！</i>"
        ),
        parse_mode="HTML",
    )


# ----------------------------------------------------
# 訊息處理主流程
# ----------------------------------------------------
async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_authorized(user.id):
        return

    user_text = update.message.text.strip()
    logger.info(f"📩 收到來自 {user.first_name} 的訊息：{user_text}")

    video_id = extract_youtube_video_id(user_text)

    if not video_id:
        await update.message.reply_text(
            "⚠️ 這似乎不是有效的 YouTube 影片連結。\n\n"
            "請傳送符合下列格式的網址：\n"
            "• <code>https://www.youtube.com/watch?v=...</code>\n"
            "• <code>https://youtu.be/...</code>",
            parse_mode="HTML",
        )
        return

    status_msg = await update.message.reply_text(
        "🔍 <b>[1/3] 正在解析影片內容...</b>", parse_mode="HTML"
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        final_transcript = get_youtube_transcript(video_id)

        if final_transcript:
            await status_msg.edit_text(
                f"📝 <b>[2/3] 取得字幕成功（約 {len(final_transcript):,} 字）</b>\n"
                "🤖 正在透過 Gemini 3.8-Flash 進行深度商業剖析...",
                parse_mode="HTML",
            )
        else:
            await status_msg.edit_text(
                "🎙 <b>[2/3] 無現成字幕，正在透過 Groq Whisper 進行高速轉錄...</b>",
                parse_mode="HTML",
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_path = download_audio_stream(user_text, tmp_dir)
                final_transcript = transcribe_audio_with_groq(audio_path)

            await status_msg.edit_text(
                f"📝 <b>[2/3] 語音轉錄完成（約 {len(final_transcript):,} 字）</b>\n"
                "🤖 正在透過 Gemini 3.8-Flash 進行深度商業剖析...",
                parse_mode="HTML",
            )

        markdown_result = generate_notes_from_text(final_transcript)
        await send_summary_and_file(update, context, markdown_result, video_id)

        try:
            await status_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"處理失敗: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ 處理過程中發生錯誤：\n<code>{html.escape(str(e))}</code>",
            parse_mode="HTML",
        )


# ----------------------------------------------------
# 主程式入口
# ----------------------------------------------------
def main():
    logger.info("⏳ 正在啟動 Telegram 機器人服務...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), message_router)
    )

    logger.info("🚀 Telegram 筆記機器人已全面上線（HTML 手機優化版）！")
    app.run_polling()


if __name__ == "__main__":
    main()