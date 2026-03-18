
import akshare as ak
import pandas as pd
import yfinance as yf
from datetime import datetime

# Your local AI setup
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:9b" # Or "qwen2.5:7b" / "deepseek-r1:7b" based on what you downloaded

def get_silver_price():
    """Fetches Silver prices with a Western Fallback if China blocks you"""
    print("📊 Fetching live Silver Futures data...")
    
    # Attempt 1: Shanghai Futures Exchange (Chinese Domestic Price)
    try:
        df_ag = ak.futures_zh_daily_sina(symbol="ag0")
        latest_price = df_ag.iloc[-1]['close']
        prev_price = df_ag.iloc[-2]['close']
        pct_change = ((latest_price - prev_price) / prev_price) * 100
        
        print(f"✅ SUCCESS: SHFE Silver Price: {latest_price} CNY/kg ({pct_change:.2f}% today)")
        return latest_price, pct_change
    except Exception as e:
        print(f"⚠️ Sina blocked the request. Falling back to Global COMEX Silver...")
        
        # Attempt 2: Global COMEX Silver Futures via Yahoo Finance (Unhackable Fallback)
        try:
            silver = yf.Ticker("SI=F")
            hist = silver.history(period="2d")
            latest_price = hist.iloc[-1]['Close']
            prev_price = hist.iloc[-2]['Close']
            pct_change = ((latest_price - prev_price) / prev_price) * 100
            
            print(f"✅ SUCCESS: COMEX Silver Price: ${latest_price:.2f} USD/oz ({pct_change:.2f}% today)")
            return latest_price, pct_change
        except Exception as e_yf:
            print(f"❌ All price fetches failed: {e_yf}")
            return 0, 0

def get_macro_news():
    """Fetches news prioritizing Eastmoney (stable) over Sina (unstable)"""
    print("\n📡 Fetching latest macro news...")
    
    # Attempt 1: Eastmoney (Usually doesn't block international IPs)
    try:
        print("Trying Eastmoney (em)...")
        df_news = ak.stock_info_global_em()
        df_news = df_news.rename(columns={'标题': 'title', '内容': 'content'})
        if not df_news.empty:
            return df_news
    except: pass

    # Attempt 2: Cailianshe (Premium wire)
    try:
        print("Trying Cailianshe (cls)...")
        df_news = ak.stock_info_global_cls(symbol="全部")
        df_news = df_news.rename(columns={'标题': 'title', '内容': 'content'})
        if not df_news.empty:
            return df_news
    except: pass

    # Attempt 3: Sina (Last resort since they block SSL)
    try:
        print("Trying Sina Finance...")
        df_news = ak.stock_info_global_sina()
        return df_news
    except:
        print("❌ All Chinese News APIs are currently blocking your IP.")
        return pd.DataFrame()

def analyze_news_with_llm(news_text, commodity="Silver"):
    """Passes the text to the Local Reasoning LLM"""
    prompt = f"""
    You are a senior commodities analyst based in Shanghai. 
    Read the following breaking news translated from Chinese: 
    "{news_text}"
    
    Analyze how this macroeconomic or industrial news affects the price of {commodity}.
    Consider supply constraints, solar panel demand, EV manufacturing, or PBOC monetary policy.
    
    You must reply in EXACTLY this format:
    SIGNAL: [BULLISH, BEARISH, or NEUTRAL]
    REASONING:[1 sentence explaining the economic logic]
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, verify=False)
        output = response.json()['response']
        
        signal = "NEUTRAL"
        if "BULLISH" in output.upper(): signal = "BULLISH"
        elif "BEARISH" in output.upper(): signal = "BEARISH"
            
        reasoning = output.split("REASONING:")[-1].strip() if "REASONING:" in output else output.strip()
        return signal, reasoning
    except Exception as e:
        return "NEUTRAL", f"LLM Error: Ensure Ollama is running in your terminal. ({e})"

def run_pro_pipeline():
    # 1. Get Quantitative Market Data
    price, pct_change = get_silver_price()
    
    # 2. Fetch News
    df_news = get_macro_news()

    if df_news.empty:
        print("\nExiting pipeline: No news available.")
        return

    # Filter for Silver/Macro keywords
    keywords =["白银", "光伏", "新能源", "央行", "降息", "美联储"] # Silver, Solar, EV, PBOC, Rate Cut, Fed
    pattern = '|'.join(keywords)
    
    # Check if 'content' or 'title' exists in the dataframe
    search_col = 'content' if 'content' in df_news.columns else df_news.columns[1]
    df_silver = df_news[df_news[search_col].astype(str).str.contains(pattern, na=False)].head(5)

    if df_silver.empty:
        print("No macro news affecting silver found right now.")
        return

    bull_score = 0
    print(f"\n🧠 AI Analyzing {len(df_silver)} Economic Drivers...")
    
    for _, row in df_silver.iterrows():
        text = str(row[search_col])
        signal, reasoning = analyze_news_with_llm(text, "Silver")
        
        if signal == "BULLISH": bull_score += 1
        elif signal == "BEARISH": bull_score -= 1
            
        print(f"\nNews: {text[:80]}...")
        print(f"Signal: {signal}")
        print(f"Logic:  {reasoning}")

    # 3. Final Synthesis Strategy
    print("\n" + "="*50)
    print("🎯 DAILY QUANTITATIVE & FUNDAMENTAL SYNTHESIS")
    print("="*50)
    print(f"Current Price Change: {pct_change:.2f}%")
    
    if bull_score > 0 and pct_change > 0:
        print("🚨 STRONG BUY: News is Bullish AND Price is confirming the trend.")
    elif bull_score < 0 and pct_change < 0:
        print("🚨 STRONG SELL: News is Bearish AND Price is dropping.")
    elif bull_score > 0 and pct_change < 0:
        print("⚠️ WATCH: News is Bullish but Price is dropping (Market has not priced in the news yet).")
    elif bull_score < 0 and pct_change > 0:
        print("⚠️ WATCH: News is Bearish but Price is rising (Potential irrational exuberance).")
    else:
        print("⚖️ HOLD: Mixed signals or neutral market.")

if __name__ == "__main__":
    run_pro_pipeline()