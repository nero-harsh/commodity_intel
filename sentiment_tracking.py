import ssl
import urllib3
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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

# Define target commodities, global tickers, and Chinese keywords
COMMODITIES = {
    "Silver": {
        "ticker": "SI=F", 
        "keywords":["白银", "光伏", "太阳能", "新能源车"]
    },
    "Gold": {
        "ticker": "GC=F", 
        "keywords":["黄金", "金价", "美联储", "避险", "央行"]
    },
    "Copper": {
        "ticker": "HG=F", 
        "keywords":["期铜", "沪铜", "房地产", "电网", "基建"]
    },
    "Aluminum": {
        "ticker": "ALI=F", 
        "keywords": ["沪铝", "电解铝", "有色金属", "产能"]
    }
}

def get_historical_technicals(ticker):
    """Fetches 1-year historical data and calculates moving averages."""
    try:
        asset = yf.Ticker(ticker)
        hist = asset.history(period="1y")
        
        if hist.empty:
            return "No historical data available."
            
        current_price = hist['Close'].iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Determine technical trend
        if current_price > sma_50 and sma_50 > sma_200:
            trend = "Strong Uptrend (Price > 50 SMA > 200 SMA)"
        elif current_price < sma_50 and sma_50 < sma_200:
            trend = "Strong Downtrend (Price < 50 SMA < 200 SMA)"
        else:
            trend = "Consolidating / Mixed Trend"
            
        return f"Current Price: {current_price:.2f}. Trend: {trend}. 50-Day MA: {sma_50:.2f}, 200-Day MA: {sma_200:.2f}."
    except Exception as e:
        return f"Error fetching historical data: {str(e)}"

def get_global_news(ticker):
    """Fetches recent global news headlines using Yahoo Finance."""
    try:
        asset = yf.Ticker(ticker)
        news_items = asset.news
        headlines = [item['title'] for item in news_items[:3]] # Keep top 3 to save LLM context
        if not headlines:
            return "No recent global news."
        return " | ".join(headlines)
    except Exception:
        return "Global news unavailable."

def get_chinese_macro_news():
    """Aggregates latest news from Eastmoney and Cailianshe."""
    news_list =[]
    
    try:
        df_em = ak.stock_info_global_em()
        if not df_em.empty:
            df_em = df_em.rename(columns={'标题': 'title', '内容': 'content'})
            news_list.append(df_em)
    except Exception:
        pass

    try:
        df_cls = ak.stock_info_global_cls(symbol="全部")
        if not df_cls.empty:
            df_cls = df_cls.rename(columns={'标题': 'title', '内容': 'content'})
            news_list.append(df_cls)
    except Exception:
        pass

    if not news_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(news_list, ignore_index=True)
    return combined_df

def analyze_commodity_with_llm(commodity, technicals, chinese_news, global_news):
    """Sends aggregated data to the LLM for a single, comprehensive analysis."""
    prompt = f"""
    You are a professional commodities quantitative researcher. Analyze the following combined data for {commodity}.

    1. 1-YEAR TECHNICAL TREND:
    {technicals}

    2. RECENT CHINESE MACROECONOMIC NEWS (Translated context required):
    {chinese_news}

    3. RECENT GLOBAL NEWS:
    {global_news}

    Synthesize the technical momentum, Chinese industrial/macro demand, and global news into a definitive forecast.
    
    You must reply strictly in the following format:
    SIGNAL:[BULLISH, BEARISH, or NEUTRAL]
    REASONING:[Provide a professional, 2 to 3 sentence justification integrating the historical trend with the fundamental news drivers.]
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2 # Lower temperature for strictly analytical, non-creative responses
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, verify=False)
        output = response.json().get('response', '')
        
        signal_line = [line for line in output.split('\n') if 'SIGNAL:' in line]
        reasoning_line = [line for line in output.split('\n') if 'REASONING:' in line]
        
        signal = signal_line[0].replace('SIGNAL:', '').strip() if signal_line else "NEUTRAL"
        reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else output.strip()
        
        return signal, reasoning
    except Exception as e:
        return "ERROR", str(e)

def main():
    print("--------------------------------------------------")
    print("COMMODITY MACRO FORECASTING SYSTEM")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------")
    
    print("[INFO] Fetching aggregate Chinese macro news...")
    df_chinese_news = get_chinese_macro_news()
    
    search_col = 'content' if 'content' in df_chinese_news.columns else (df_chinese_news.columns[1] if not df_chinese_news.empty else None)
    
    results =[]

    for commodity, data in COMMODITIES.items():
        print(f"\n[INFO] Processing data for {commodity}...")
        
        # 1. Get Technicals
        technicals = get_historical_technicals(data["ticker"])
        
        # 2. Get Global News
        global_news = get_global_news(data["ticker"])
        
        # 3. Filter Chinese News
        chinese_context = "No relevant Chinese news found today."
        if not df_chinese_news.empty and search_col:
            pattern = '|'.join(data["keywords"])
            filtered_news = df_chinese_news[df_chinese_news[search_col].astype(str).str.contains(pattern, na=False)].head(3)
            if not filtered_news.empty:
                chinese_context = " | ".join(filtered_news[search_col].astype(str).tolist())
        
        # 4. LLM Inference
        print(f"[INFO] Executing LLM inference for {commodity}...")
        signal, reasoning = analyze_commodity_with_llm(commodity, technicals, chinese_context, global_news)
        
        results.append({
            "Commodity": commodity,
            "Signal": signal,
            "Reasoning": reasoning,
            "Technicals": technicals.split('.')[1].strip() if '.' in technicals else technicals
        })

    # Print Final Professional Report
    print("\n\n" + "="*80)
    print("FINAL COMMODITY FORECAST REPORT")
    print("="*80)
    
    for res in results:
        print(f"COMMODITY : {res['Commodity']}")
        print(f"SIGNAL    : {res['Signal']}")
        print(f"TREND     : {res['Technicals']}")
        print(f"REASONING : {res['Reasoning']}")
        print("-" * 80)

if __name__ == "__main__":
    main()