import os
import ssl
import urllib3
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
from deep_translator import GoogleTranslator

# Disable SSL Warnings for Chinese APIs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:9b"
LLM_TEMPERATURE = 0.5 
MAX_NEWS_ITEMS = 15  # Configurable parameter to control news ingestion volume

# File Paths
CSV_LOG_FILE = "commodity_data_matrix.csv"
TXT_LOG_FILE = "forecast_reasoning_log.txt"

COMMODITIES = {
    "Silver": {
        "ticker": "SI=F", 
        "keywords": ["白银", "银价", "光伏", "太阳能", "新能源车", "电子"]
    },
    "Gold": {
        "ticker": "GC=F", 
        "keywords":["黄金", "金价", "美联储", "避险", "央行", "地缘政治"]
    },
    "Copper": {
        "ticker": "HG=F", 
        "keywords":["期铜", "沪铜", "铜价", "房地产", "电网", "基建"]
    },
    "Aluminum": {
        "ticker": "ALI=F", 
        "keywords": ["沪铝", "电解铝", "有色金属", "产能", "电力"]
    },
    "Lithium": {
        "ticker": "ALB", 
        "keywords":["碳酸锂", "锂价", "电池", "电动车", "新能源汽车", "宁德时代"]
    }
}

def get_historical_technicals_2y(ticker):
    """Fetches 2-year historical data, SMAs, daily (1mo), and monthly (24mo) prices."""
    try:
        asset = yf.Ticker(ticker)
        hist = asset.history(period="2y")
        
        if hist.empty:
            return "No historical data available."
            
        current_price = hist['Close'].iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Last 20 trading days (approx 1 month)
        last_month_daily = hist['Close'].tail(20).round(2)
        daily_str = ", ".join([f"{idx.strftime('%m-%d')}: {val}" for idx, val in last_month_daily.items()])
        
        # Last 24 months closing prices
        monthly_closes = hist['Close'].resample('ME').last().round(2).tail(24)
        monthly_str = ", ".join([f"{idx.strftime('%Y-%m')}: {val}" for idx, val in monthly_closes.items()])
        
        technicals_payload = (
            f"Current Price: {current_price:.2f}\n"
            f"50-Day SMA: {sma_50:.2f} | 200-Day SMA: {sma_200:.2f}\n"
            f"Daily Closes (Last 20 Days): {daily_str}\n"
            f"Monthly Closes (Last 24 Months): {monthly_str}"
        )
        return technicals_payload
    except Exception as e:
        return f"Error fetching historical data: {str(e)}"

def get_global_news(ticker):
    """Fetches recent global geopolitical and market news."""
    try:
        asset = yf.Ticker(ticker)
        news_items = asset.news
        headlines = [item['title'] for item in news_items[:5]]
        if not headlines:
            return "No recent global news found."
        return " | ".join(headlines)
    except Exception:
        return "Global news unavailable."

def get_chinese_macro_news():
    """Aggregates latest news from Eastmoney, Cailianshe, and Sina Finance."""
    import akshare as ak
    news_list = []
    sources =[
        ("Eastmoney", ak.stock_info_global_em),
        ("Cailianshe", lambda: ak.stock_info_global_cls(symbol="全部")),
        ("Sina Finance", ak.stock_info_global_sina)
    ]
    
    print("[INFO] Initiating connection to Chinese financial terminals...")
    
    for source_name, fetch_func in tqdm(sources, desc="Fetching Asian Macro Data", unit="source"):
        try:
            df = fetch_func()
            if not df.empty:
                if '标题' in df.columns and '内容' in df.columns:
                    df = df.rename(columns={'标题': 'title', '内容': 'content'})
                elif 'title' in df.columns and 'content' in df.columns:
                    pass
                else:
                    continue
                news_list.append(df[['title', 'content']])
        except Exception:
            pass 

    if not news_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(news_list, ignore_index=True)
    return combined_df

def translate_news_for_csv(chinese_news_list):
    """Translates a list of Chinese news strings to English for CSV storage."""
    translator = GoogleTranslator(source='zh-CN', target='en')
    translated_list =[]
    for text in chinese_news_list:
        try:
            # Truncate to 4500 chars to avoid Google Translate API limits
            translated_list.append(translator.translate(text[:4500]))
        except Exception:
            translated_list.append("Translation Failed / API Limit Reached")
    return " | ".join(translated_list)

def analyze_commodity_with_llm(commodity, technicals, chinese_news_raw, global_news):
    """Sends aggregated data to the LLM for multi-horizon forecasting."""
    prompt = f"""
    You are a professional quantitative commodities researcher. 
    Analyze the following combined data for {commodity} to forecast price movements across multiple time horizons.

    1. 2-YEAR PRICE HISTORY & TECHNICALS:
    {technicals}

    2. RAW CHINESE MACROECONOMIC & INDUSTRIAL NEWS (HIGH WEIGHT):
    {chinese_news_raw}

    3. GLOBAL MACRO & GEOPOLITICAL NEWS (LOW WEIGHT):
    {global_news}

    INSTRUCTIONS:
    - Evaluate short-term momentum using the 20-day daily prices and recent news.
    - Evaluate long-term trends using the 24-month prices, 200-SMA, and structural industrial shifts (e.g. EV demand, solar capacity).
    - Chinese industrial data takes precedence over global geopolitical noise.
    
    You must reply strictly in the following format. Do not add markdown blocks, just the text:
    1_WEEK_SIGNAL: [BULLISH, BEARISH, or NEUTRAL]
    1_MONTH_SIGNAL: [BULLISH, BEARISH, or NEUTRAL]
    3_MONTH_SIGNAL: [BULLISH, BEARISH, or NEUTRAL]
    6_MONTH_SIGNAL:[BULLISH, BEARISH, or NEUTRAL]
    REASONING:[Provide a comprehensive, multi-paragraph professional analysis explaining the short-term catalysts and long-term structural drivers.]
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, verify=False)
        if response.status_code != 200:
            return {"error": f"LLM API returned status {response.status_code}"}
            
        output = response.json().get('response', '')
        
        # Parse the structured output
        signals = {
            "1_Week": "NEUTRAL",
            "1_Month": "NEUTRAL",
            "3_Month": "NEUTRAL",
            "6_Month": "NEUTRAL",
            "Reasoning": ""
        }
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith("1_WEEK_SIGNAL:"): signals["1_Week"] = line.replace("1_WEEK_SIGNAL:", "").strip()
            elif line.startswith("1_MONTH_SIGNAL:"): signals["1_Month"] = line.replace("1_MONTH_SIGNAL:", "").strip()
            elif line.startswith("3_MONTH_SIGNAL:"): signals["3_Month"] = line.replace("3_MONTH_SIGNAL:", "").strip()
            elif line.startswith("6_MONTH_SIGNAL:"): signals["6_Month"] = line.replace("6_MONTH_SIGNAL:", "").strip()
            
        if "REASONING:" in output:
            signals["Reasoning"] = output[output.find("REASONING:") + 10:].strip()
        else:
            signals["Reasoning"] = output.strip()
            
        return signals
    except Exception as e:
        return {"error": f"Failed to connect to local LLM: {str(e)}"}

def save_to_csv(data_dict):
    """Appends the raw data matrix to a CSV file."""
    df = pd.DataFrame([data_dict])
    file_exists = os.path.isfile(CSV_LOG_FILE)
    df.to_csv(CSV_LOG_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')

def save_to_txt(commodity, signals):
    """Appends the multi-horizon forecast and reasoning to a TXT log."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(TXT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] COMMODITY: {commodity}\n")
        f.write(f"1 WEEK FORECAST   : {signals.get('1_Week', 'ERROR')}\n")
        f.write(f"1 MONTH FORECAST  : {signals.get('1_Month', 'ERROR')}\n")
        f.write(f"3 MONTH FORECAST  : {signals.get('3_Month', 'ERROR')}\n")
        f.write(f"6 MONTH FORECAST  : {signals.get('6_Month', 'ERROR')}\n")
        f.write(f"REASONING:\n{signals.get('Reasoning', 'ERROR')}\n")
        f.write("-" * 80 + "\n\n")

def main():
    print("--------------------------------------------------")
    print("COMMODITY MACRO FORECASTING SYSTEM (MULTI-HORIZON)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------")
    
    df_chinese_news = get_chinese_macro_news()
    if df_chinese_news.empty:
        print("[ERROR] Failed to fetch Chinese macro news. Ensure network connectivity.")
    else:
        print(f"[INFO] Successfully aggregated {len(df_chinese_news)} financial alerts.")

    print("\n[INFO] Commencing Data Extraction and LLM Inference Pipeline...")
    
    pbar = tqdm(COMMODITIES.items(), desc="Analyzing Markets", unit="commodity")
    
    for commodity, data in pbar:
        pbar.set_description(f"Analyzing {commodity}")
        
        # 1. Fetch 2-Year Technicals & Prices
        technicals = get_historical_technicals_2y(data["ticker"])
        
        # 2. Fetch Global News
        global_news = get_global_news(data["ticker"])
        
        # 3. Filter and Limit Chinese News (Top-K parameter logic)
        chinese_news_raw_list =[]
        if not df_chinese_news.empty:
            pattern = '|'.join(data["keywords"])
            filtered_news = df_chinese_news[df_chinese_news['content'].astype(str).str.contains(pattern, na=False)]
            # Apply the configurable limit
            filtered_news = filtered_news.head(MAX_NEWS_ITEMS)
            if not filtered_news.empty:
                chinese_news_raw_list = filtered_news['content'].astype(str).tolist()
                
        chinese_context_raw = " | ".join(chinese_news_raw_list) if chinese_news_raw_list else "No relevant Chinese industrial news found."
        
        # 4. Run LLM Inference (Feeding RAW Chinese text)
        signals = analyze_commodity_with_llm(commodity, technicals, chinese_context_raw, global_news)
        
        # 5. Translate the ingested Chinese news strictly for CSV readability
        translated_chinese_news = "No news to translate."
        if chinese_news_raw_list:
            pbar.set_description(f"Translating {commodity} logs")
            translated_chinese_news = translate_news_for_csv(chinese_news_raw_list)
        
        # 6. Save Data to CSV Matrix
        csv_data = {
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Commodity": commodity,
            "1_Week_Signal": signals.get("1_Week", "ERROR"),
            "1_Month_Signal": signals.get("1_Month", "ERROR"),
            "3_Month_Signal": signals.get("3_Month", "ERROR"),
            "6_Month_Signal": signals.get("6_Month", "ERROR"),
            "Technicals_Context": technicals,
            "Raw_Chinese_News_Fed_To_LLM": chinese_context_raw,
            "Translated_Chinese_News": translated_chinese_news,
            "Global_News_Context": global_news
        }
        save_to_csv(csv_data)
        
        # 7. Save Reasoning to TXT Log
        save_to_txt(commodity, signals)

    print("\n--------------------------------------------------")
    print("[SUCCESS] Pipeline Execution Completed.")
    print(f"[INFO] Raw data and English translations appended to: {CSV_LOG_FILE}")
    print(f"[INFO] Multi-horizon forecast reasoning appended to: {TXT_LOG_FILE}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()