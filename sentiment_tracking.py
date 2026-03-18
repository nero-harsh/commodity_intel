import akshare as ak
import pandas as pd
import torch
from deep_translator import GoogleTranslator
from transformers import pipeline
import time
from datetime import datetime
import warnings

# Suppress benign AI model warnings
warnings.filterwarnings("ignore")

# 1. Hardware Acceleration (Mac M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f" Running AI Engine on: {device.upper()}")

# 2. Load the Chinese Financial AI Model
print("Loading Chinese Financial Sentiment Model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2", 
    device=device
)

# 3. Expanded Scope: Commodities and their Dependent Industries
COMMODITY_MAPPING = {
    "Silver (白银)": ["白银", "银价", "光伏", "太阳能", "新能源车"], 
    # Silver relies heavily on solar panels (光伏) and EVs (新能源车)
    
    "Gold (黄金)": ["黄金", "金价", "贵金属", "避险", "美联储降息"], 
    # Gold is driven by safe-haven (避险) and Fed rates (美联储)
    
    "Copper (铜)":["期铜", "沪铜", "铜价", "房地产", "电网", "基建"], 
    # Copper is driven by Chinese real estate (房地产) and grid infrastructure
    
    "Lithium/Battery (锂)":["碳酸锂", "锂价", "电池", "电动车", "宁德时代"],
    # Lithium is driven by EV batteries and giants like CATL (宁德时代)
    
    "Rare Earths (稀土)": ["稀土", "磁材", "出口管制"]
    # Rare earths are driven by magnetic materials and export controls
}

def fetch_all_sources():
    """Fetches rolling macro news from all 3 major Chinese financial hubs"""
    all_news =[]
    
    print("📡 Fetching Eastmoney (东方财富)...")
    try:
        df_em = ak.stock_info_global_em()
        for _, row in df_em.iterrows():
            text = str(row.get('标题', '')) + " " + str(row.get('内容', ''))
            dt = row.get('时间', row.get('发布时间', datetime.now()))
            all_news.append({"Source": "Eastmoney", "Date": dt, "Chinese_Text": text})
    except Exception as e: print(f"⚠️ Eastmoney error: {e}")

    print(" Fetching Sina Finance (新浪财经)...")
    try:
        df_sina = ak.stock_info_global_sina()
        for _, row in df_sina.iterrows():
            text = str(row.get('title', '')) + " " + str(row.get('content', ''))
            dt = row.get('create_time', row.get('时间', datetime.now()))
            all_news.append({"Source": "Sina", "Date": dt, "Chinese_Text": text})
    except Exception as e: print(f"⚠️ Sina error: {e}")

    print("📡 Fetching Cailianshe (财联社)...")
    try:
        try:
            df_cls = ak.stock_info_global_cls(symbol="全部")
        except:
            df_cls = ak.stock_info_global_cls() # Fallback for older akshare versions
        for _, row in df_cls.iterrows():
            text = str(row.get('标题', '')) + " " + str(row.get('内容', ''))
            dt = row.get('发布时间', row.get('时间', datetime.now()))
            all_news.append({"Source": "Cailianshe", "Date": dt, "Chinese_Text": text})
    except Exception as e: print(f"⚠️ Cailianshe error: {e}")

    return pd.DataFrame(all_news)

def parse_and_filter_dates(df):
    """Standardizes dates and filters for the last 7 days"""
    def clean_date(d):
        try:
            return pd.to_datetime(str(d))
        except:
            return pd.Timestamp.now()
            
    df['Date'] = df['Date'].apply(clean_date)
    # Fix timestamps that lack years (some APIs return just "10:30" resulting in 1900)
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=datetime.now().year) if x.year == 1900 else x)
    
    # Keep only last 7 days
    seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    df = df[df['Date'] >= seven_days_ago]
    
    # Drop duplicated news spanning across platforms
    df = df.drop_duplicates(subset=['Chinese_Text'])
    return df

def run_analysis():
    raw_df = fetch_all_sources()
    
    if raw_df.empty:
        print("❌ Could not fetch data. Your IP might be temporarily blocked. Trying historical futures fallback...")
        try:
            raw_df = ak.futures_news_shfe() # Shanghai Futures Exchange historical news
            raw_df = raw_df.rename(columns={'datetime': 'Date', 'title': 'Chinese_Text'})
            raw_df['Source'] = 'SHFE Historical'
        except:
            print("Fallback failed. Please try again in an hour.")
            return

    clean_df = parse_and_filter_dates(raw_df)
    print(f"\n✅ Aggregated {len(clean_df)} total market news articles from the last 7 days.")
    
    # Map news to Commodities
    matched_news =[]
    for _, row in clean_df.iterrows():
        text = row['Chinese_Text']
        if len(text) < 10: continue
            
        tags =[]
        for comm, keywords in COMMODITY_MAPPING.items():
            if any(kw in text for kw in keywords):
                tags.append(comm)
                
        if tags:
            matched_news.append({
                "Date": row['Date'],
                "Source": row['Source'],
                "Commodities": ", ".join(tags),
                "Chinese_Text": text
            })

    df_matched = pd.DataFrame(matched_news)
    
    if df_matched.empty:
        print("📉 No specific commodity-related news found in this 7-day window.")
        return

    print(f"🔍 Found {len(df_matched)} highly relevant news items! Translating & analyzing sentiment...")
    
    results =[]
    commodity_scores = {c: 0 for c in COMMODITY_MAPPING.keys()}
    translator = GoogleTranslator(source='zh-CN', target='en')

    for index, row in df_matched.iterrows():
        text = row['Chinese_Text']
        
        # 1. AI Sentiment
        try:
            sent = sentiment_pipeline(text[:512])[0]
            label = sent['label']
        except:
            label = "Neutral"
            
        # 2. Translation (with truncation to avoid API errors)
        try:
            eng_text = translator.translate(text[:800])
        except Exception:
            eng_text = "Translation API timeout."
            
        results.append({
            "Date": row['Date'].strftime('%Y-%m-%d %H:%M'),
            "Commodities_Affected": row['Commodities'],
            "Source": row['Source'],
            "AI_Sentiment": label,
            "English_News": eng_text,
            "Chinese_News": text
        })
        
        # 3. Update Scores
        for comm in row['Commodities'].split(", "):
            if label == "Positive": commodity_scores[comm] += 1
            elif label == "Negative": commodity_scores[comm] -= 1
            
        time.sleep(0.5) # Prevent translation API block

    # Export to CSV
    df_results = pd.DataFrame(results)
    df_results.sort_values(by='Date', ascending=False, inplace=True)
    df_results.to_csv("commodity_news_analysis.csv", index=False, encoding='utf-8-sig')
    
    # Print Forecast Report
    print("\n" + "="*50)
    print("📈 MULTI-COMMODITY 7-DAY FORECAST (CHINA MACRO)")
    print("="*50)
    
    log_content = f"\nForecast Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    for comm, score in commodity_scores.items():
        if score > 0: forecast = f"🟢 BULLISH (Score: +{score})"
        elif score < 0: forecast = f"🔴 BEARISH (Score: {score})"
        else: forecast = "🟡 NEUTRAL (Score: 0)"
        
        report_line = f"{comm.ljust(35)} | {forecast}"
        print(report_line)
        log_content += report_line + "\n"
        
    print("\n✅ Full translation and analysis saved to 'commodity_news_analysis.csv'")
    
    with open("forecast_log.txt", "a", encoding='utf-8') as f:
        f.write(log_content + "-"*50 + "\n")

if __name__ == "__main__":
    run_analysis()