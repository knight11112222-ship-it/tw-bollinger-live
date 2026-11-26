import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TW_TZ = ZoneInfo("Asia/Taipei")
ORDINARY_STOCK_RE = re.compile(r"^[1-9]\d{3}$")

CONFIG = {
    "FINMIND_TOKEN": os.getenv("FINMIND_TOKEN", "").strip(),
    "FINMIND_URL": "https://api.finmindtrade.com/api/v4/data",
    "TWSE_OPENAPI_URL": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "TPEX_OPENAPI_URL": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",

    # 撈取門檻：納入成交值 >= 3 億元 (前端預設 5 億，拉寬底層池讓使用者可在網頁上調低至 3 億測試)
    "FETCH_MIN_TRADING_MONEY": 300_000_000,
    "MAX_STOCKS": 160,                 # 依成交值排序最多保留前 160 檔，確保 FinMind 免費版每小時 600 次額度充裕
    "EXCLUDE_ETF": True,

    # 各週期保留 K 棒數量 (兼顧 60 期布林計算與 JSON 體積)
    "KEEP_BARS_W": 65,
    "KEEP_BARS_D": 95,
    "KEEP_BARS_60M": 85,
    "KEEP_BARS_30M": 95,

    "FINMIND_SLEEP_SEC": 0.14,
    "YFINANCE_SLEEP_SEC": 0.18,
    "OUTPUT_FILE": "data.json",
    "FORCE_FULL_REFRESH": os.getenv("FORCE_FULL_REFRESH", "false").lower() == "true",
}


# ==============================================================================
# 1. 官方免費 OpenAPI 快篩 (TWSE 上市 + TPEx 上櫃)
# ==============================================================================
def parse_roc_date(raw_date: str) -> str:
    cleaned = re.sub(r"\D", "", str(raw_date or "").strip())
    if len(cleaned) == 7:
        return f"{int(cleaned[:3]) + 1911:04d}-{cleaned[3:5]}-{cleaned[5:7]}"
    if len(cleaned) == 8:
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return ""


def parse_number(raw_val) -> float:
    if raw_val is None:
        return np.nan
    s = str(raw_val).strip().replace(",", "")
    if not s or s in ("-", "--", "---", "----", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def fetch_liquid_stocks(min_money: float, max_stocks: int, exclude_etf: bool = True) -> tuple[list[dict], dict]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    records = []
    market_dates = {"twse": "", "tpex": ""}

    # 1) TWSE 上市
    resp_twse = requests.get(CONFIG["TWSE_OPENAPI_URL"], headers=headers, timeout=25)
    resp_twse.raise_for_status()
    for row in resp_twse.json():
        sid = str(row.get("Code", "")).strip()
        if exclude_etf and not ORDINARY_STOCK_RE.match(sid):
            continue
        d_str = parse_roc_date(row.get("Date", ""))
        if d_str and d_str > market_dates["twse"]:
            market_dates["twse"] = d_str
        trade_val = parse_number(row.get("TradeValue"))
        close_px = parse_number(row.get("ClosingPrice"))
        if np.isnan(trade_val) or np.isnan(close_px) or trade_val < min_money:
            continue
        records.append({
            "id": sid,
            "name": str(row.get("Name", "")).strip(),
            "market": "twse",
            "valYi": round(trade_val / 1e8, 2),
            "openapiClose": round(close_px, 2),
            "openapiDate": d_str,
        })

    # 2) TPEx 上櫃
    resp_tpex = requests.get(CONFIG["TPEX_OPENAPI_URL"], headers=headers, timeout=25)
    resp_tpex.raise_for_status()
    for row in resp_tpex.json():
        sid = str(row.get("SecuritiesCompanyCode", "")).strip()
        if exclude_etf and not ORDINARY_STOCK_RE.match(sid):
            continue
        d_str = parse_roc_date(row.get("Date", ""))
        if d_str and d_str > market_dates["tpex"]:
            market_dates["tpex"] = d_str
        trade_val = parse_number(row.get("TransactionAmount"))
        close_px = parse_number(row.get("Close"))
        if np.isnan(trade_val) or np.isnan(close_px) or trade_val < min_money:
            continue
        records.append({
            "id": sid,
            "name": str(row.get("CompanyName", "")).strip(),
            "market": "tpex",
            "valYi": round(trade_val / 1e8, 2),
            "openapiClose": round(close_px, 2),
            "openapiDate": d_str,
        })

    records.sort(key=lambda x: x["valYi"], reverse=True)
    return records[:max_stocks], market_dates


# ==============================================================================
# 2. FinMind 免費 API 客戶端
# ==============================================================================
class FinMindFreeClient:
    def __init__(self, token: str, url: str):
        self.url = url
        self.request_count = 0
        self.rate_limited = False
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def get_dataset(self, dataset: str, data_id: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        if self.rate_limited:
            return pd.DataFrame()
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        resp = self.session.get(self.url, params=params, timeout=25)
        self.request_count += 1
        if resp.status_code in (402, 429):
            print(f"⚠️ FinMind 達到每小時請求上限 (HTTP {resp.status_code})，後續將保留已取得資料。")
            self.rate_limited = True
            return pd.DataFrame()
        if resp.status_code == 403:
            return pd.DataFrame()
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != 200:
            return pd.DataFrame()
        data = payload.get("data", [])
        return pd.DataFrame(data) if data else pd.DataFrame()


def fetch_industry_map(client: FinMindFreeClient) -> dict:
    try:
        df = client.get_dataset("TaiwanStockInfo")
        if not df.empty and "stock_id" in df.columns and "industry_category" in df.columns:
            df = df.drop_duplicates(subset=["stock_id"])
            return dict(zip(df["stock_id"].astype(str), df["industry_category"].astype(str)))
    except Exception as e:
        print(f"ℹ️ 無法取得產業分類表，將改用市場別標示: {e}")
    return {}


# ==============================================================================
# 3. 資料轉換與緊湊序列化工具
# ==============================================================================
def daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df = df.set_index("dt").sort_index()
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    weekly = weekly.reset_index()
    weekly["date"] = weekly["dt"].dt.strftime("%Y-%m-%d")
    return weekly


def df_daily_to_compact_bars(df: pd.DataFrame, keep_n: int) -> tuple[list, list]:
    """輸出 [[ts_ms, open, high, low, close, volume_lots], ...] 與對應 YYYY-MM-DD 清單"""
    sub = df.tail(keep_n).copy()
    bars = []
    dates = []
    for _, r in sub.iterrows():
        d_str = str(r["date"])[:10]
        dt = datetime.strptime(d_str, "%Y-%m-%d").replace(hour=13, minute=30, tzinfo=TW_TZ)
        ts_ms = int(dt.timestamp() * 1000)
        vol_lots = int(round(float(r["volume"]) / 1000.0)) if float(r["volume"]) > 50000 else int(round(float(r["volume"])))
        bars.append([
            ts_ms,
            round(float(r["open"]), 2),
            round(float(r["high"]), 2),
            round(float(r["low"]), 2),
            round(float(r["close"]), 2),
            max(0, vol_lots),
        ])
        dates.append(d_str)
    return bars, dates


def fetch_yfinance_intraday_compact(stock_id: str, market: str, freq: str, keep_n: int) -> list:
    suffix = ".TWO" if market == "tpex" else ".TW"
    interval = "30m" if freq == "30m" else "60m"
    period = "1mo" if freq == "30m" else "2mo"
    try:
        hist = yf.Ticker(f"{stock_id}{suffix}").history(period=period, interval=interval, auto_adjust=False)
        if hist.empty:
            return []
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.reset_index()
        time_col = "Datetime" if "Datetime" in hist.columns else hist.columns[0]
        hist = hist.rename(columns={
            time_col: "dt", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        dt_series = pd.to_datetime(hist["dt"])
        if dt_series.dt.tz is None:
            dt_series = dt_series.dt.tz_localize(TW_TZ)
        else:
            dt_series = dt_series.dt.tz_convert(TW_TZ)
        hist["dt"] = dt_series
        hist = hist.dropna(subset=["close"])
        hist = hist[hist["close"] > 0].sort_values("dt").tail(keep_n)

        bars = []
        for _, r in hist.iterrows():
            ts_ms = int(r["dt"].timestamp() * 1000)
            vol_lots = int(round(float(r["volume"]) / 1000.0))
            bars.append([
                ts_ms,
                round(float(r["open"]), 2),
                round(float(r["high"]), 2),
                round(float(r["low"]), 2),
                round(float(r["close"]), 2),
                max(0, vol_lots),
            ])
        return bars
    except Exception:
        return []


def fetch_institutional_series(client: FinMindFreeClient, stock_id: str, daily_dates: list[str], start_date: str, end_date: str) -> dict:
    """回傳對齊 daily_dates 之三大法人每日買賣超張數 [[foreign, trust, dealer], ...] 與統計值"""
    empty_series = [[0, 0, 0] for _ in daily_dates]
    default_out = {
        "instDaily": empty_series,
        "foreign5dLots": 0,
        "trust5dLots": 0,
        "dealer5dLots": 0,
        "trustDays": 0,
    }
    if client.rate_limited or not daily_dates:
        return default_out

    try:
        df = client.get_dataset("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date, end_date)
        time.sleep(CONFIG["FINMIND_SLEEP_SEC"])
        if df.empty:
            return default_out

        df["net_lots"] = (pd.to_numeric(df["buy"], errors="coerce").fillna(0) - pd.to_numeric(df["sell"], errors="coerce").fillna(0)) / 1000.0
        # 歸類外資、投信、自營
        def _cat(name):
            if name in ("Foreign_Investor", "Foreign_Dealer_Self"):
                return "F"
            if name == "Investment_Trust":
                return "T"
            return "D"

        df["cat"] = df["name"].map(_cat)
        piv = df.pivot_table(index="date", columns="cat", values="net_lots", aggfunc="sum", fill_value=0.0)
        for col in ("F", "T", "D"):
            if col not in piv.columns:
                piv[col] = 0.0

        aligned = []
        for d in daily_dates:
            if d in piv.index:
                r = piv.loc[d]
                aligned.append([int(round(r["F"])), int(round(r["T"])), int(round(r["D"]))])
            else:
                aligned.append([0, 0, 0])

        # 找出最近有法人數據的 5 個交易日
        valid_dates = sorted(piv.index.tolist())
        last5 = piv.loc[valid_dates[-5:]] if valid_dates else pd.DataFrame(columns=["F", "T", "D"])
        f5 = int(round(last5["F"].sum())) if not last5.empty else 0
        t5 = int(round(last5["T"].sum())) if not last5.empty else 0
        d5 = int(round(last5["D"].sum())) if not last5.empty else 0

        # 計算投信連續買超天數
        trust_streak = 0
        for d in reversed(valid_dates):
            if piv.loc[d, "T"] > 0:
                trust_streak += 1
            else:
                break

        return {
            "instDaily": aligned,
            "foreign5dLots": f5,
            "trust5dLots": t5,
            "dealer5dLots": d5,
            "trustDays": trust_streak,
        }
    except Exception:
        return default_out


# ==============================================================================
# 4. 智慧快取判斷：盤中重複執行只更新 yfinance 分K，節省 FinMind 免費額度
# ==============================================================================
def load_existing_cache(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    now_tw = datetime.now(TW_TZ)
    print("=" * 72)
    print(f"🚀 開始更新台股真實布林通道資料 ({now_tw.strftime('%Y-%m-%d %H:%M:%S CST')})")
    print("=" * 72)

    liquid_stocks, market_dates = fetch_liquid_stocks(
        min_money=CONFIG["FETCH_MIN_TRADING_MONEY"],
        max_stocks=CONFIG["MAX_STOCKS"],
        exclude_etf=CONFIG["EXCLUDE_ETF"],
    )
    valid_dates = [d for d in market_dates.values() if d]
    latest_market_date = max(valid_dates) if valid_dates else now_tw.strftime("%Y-%m-%d")
    print(f"📊 TWSE日: {market_dates['twse']} | TPEx日: {market_dates['tpex']} | 篩出 >= 3億普通股: {len(liquid_stocks)} 檔")

    existing = load_existing_cache(CONFIG["OUTPUT_FILE"])
    cached_by_id = {}
    can_reuse_daily = (
        not CONFIG["FORCE_FULL_REFRESH"]
        and existing.get("meta", {}).get("marketDate") == latest_market_date
        and len(existing.get("stocks", [])) >= 30
    )

    if can_reuse_daily:
        # 額外檢查：若已過 16:00 CST 且上次更新在 16:00 前，則強制重抓以補齊今日盤後三大法人
        prev_ts = existing.get("meta", {}).get("updatedAt", "")
        if now_tw.hour >= 16 and prev_ts[:10] == now_tw.strftime("%Y-%m-%d") and prev_ts[11:16] < "16:00":
            print("🕓 已過 16:00 法人揭露時間，執行完整更新補齊最新三大法人籌碼...")
            can_reuse_daily = False
        else:
            cached_by_id = {s["id"]: s for s in existing.get("stocks", [])}
            print(f"⚡ 盤中快取命中 (日線日期 {latest_market_date} 未變動)：重用日/週線與法人，僅即時刷新 yfinance 30m/60m 分K！")

    client = FinMindFreeClient(CONFIG["FINMIND_TOKEN"], CONFIG["FINMIND_URL"])
    industry_map = {} if can_reuse_daily else fetch_industry_map(client)

    daily_start = (now_tw - timedelta(days=430)).strftime("%Y-%m-%d")
    inst_start = (now_tw - timedelta(days=150)).strftime("%Y-%m-%d")
    query_end = (now_tw + timedelta(days=1)).strftime("%Y-%m-%d")

    output_stocks = []

    for idx, item in enumerate(liquid_stocks, 1):
        sid = item["id"]
        market = item["market"]

        try:
            # --- A. 取得或重用日線、週線與三大法人 ---
            if can_reuse_daily and sid in cached_by_id:
                rec = dict(cached_by_id[sid])
                bars_w = rec["bars"]["W"]
                bars_d = rec["bars"]["D"]
                sector = rec.get("sector", market.upper())
                inst_info = {
                    "instDaily": rec.get("instDaily", []),
                    "foreign5dLots": rec.get("foreign5dLots", 0),
                    "trust5dLots": rec.get("trust5dLots", 0),
                    "dealer5dLots": rec.get("dealer5dLots", 0),
                    "trustDays": rec.get("trustDays", 0),
                }
            else:
                df_d = client.get_dataset("TaiwanStockPrice", sid, daily_start, query_end)
                time.sleep(CONFIG["FINMIND_SLEEP_SEC"])
                if df_d.empty or len(df_d) < 30:
                    continue

                df_d = df_d.rename(columns={
                    "max": "high",
                    "min": "low",
                    "Trading_Volume": "volume",
                    "Trading_money": "trading_money",
                }).sort_values("date").reset_index(drop=True)
                for col in ("open", "high", "low", "close", "volume"):
                    df_d[col] = pd.to_numeric(df_d[col], errors="coerce")
                df_d = df_d.dropna(subset=["close"])
                df_d = df_d[df_d["close"] > 0].reset_index(drop=True)
                if len(df_d) < 25:
                    continue

                # 若 TWSE/TPEx OpenAPI 日期比 FinMind 最新日線更新，先用 OpenAPI 收盤價補足，避免股價差一天
                last_fm_date = str(df_d.iloc[-1]["date"])[:10]
                if item["openapiDate"] and item["openapiDate"] > last_fm_date and item["openapiClose"] > 0:
                    df_d = pd.concat([df_d, pd.DataFrame([{
                        "date": item["openapiDate"],
                        "open": item["openapiClose"],
                        "high": item["openapiClose"],
                        "low": item["openapiClose"],
                        "close": item["openapiClose"],
                        "volume": 0,
                    }])], ignore_index=True)

                df_w = daily_to_weekly(df_d)
                bars_d, daily_dates = df_daily_to_compact_bars(df_d, CONFIG["KEEP_BARS_D"])
                bars_w, _ = df_daily_to_compact_bars(df_w, CONFIG["KEEP_BARS_W"])
                sector = industry_map.get(sid, "上市" if market == "twse" else "上櫃")

                inst_info = fetch_institutional_series(client, sid, daily_dates, inst_start, query_end)

            # --- B. 即時刷新 yfinance 30m / 60m 分K ---
            bars_30m = fetch_yfinance_intraday_compact(sid, market, "30m", CONFIG["KEEP_BARS_30M"])
            time.sleep(CONFIG["YFINANCE_SLEEP_SEC"])
            bars_60m = fetch_yfinance_intraday_compact(sid, market, "60m", CONFIG["KEEP_BARS_60M"])
            time.sleep(CONFIG["YFINANCE_SLEEP_SEC"])

            # 決定最新即時價格與昨收基準
            latest_daily_close = bars_d[-1][4]
            prev_daily_close = bars_d[-2][4] if len(bars_d) >= 2 else latest_daily_close
            latest_price = latest_daily_close
            prev_close = prev_daily_close

            if bars_30m:
                last_30m_dt = datetime.fromtimestamp(bars_30m[-1][0] / 1000.0, tz=TW_TZ).strftime("%Y-%m-%d")
                last_d_dt = datetime.fromtimestamp(bars_d[-1][0] / 1000.0, tz=TW_TZ).strftime("%Y-%m-%d")
                if last_30m_dt > last_d_dt:
                    # 盤中最新價：昨收為最新完整日線收盤
                    latest_price = bars_30m[-1][4]
                    prev_close = latest_daily_close
                elif last_30m_dt == last_d_dt:
                    latest_price = bars_30m[-1][4]

            total_5d_lots = inst_info["foreign5dLots"] + inst_info["trust5dLots"] + inst_info["dealer5dLots"]
            inst_5d_yi = round((total_5d_lots * 1000.0 * latest_price) / 1e8, 2)

            output_stocks.append({
                "id": sid,
                "name": item["name"],
                "market": market,
                "sector": sector,
                "valYi": item["valYi"],
                "price": latest_price,
                "prevClose": prev_close,
                "foreign5dLots": inst_info["foreign5dLots"],
                "trust5dLots": inst_info["trust5dLots"],
                "dealer5dLots": inst_info["dealer5dLots"],
                "inst5dYi": inst_5d_yi,
                "trustDays": inst_info["trustDays"],
                "instDaily": inst_info["instDaily"],
                "bars": {
                    "W": bars_w,
                    "D": bars_d,
                    "60m": bars_60m,
                    "30m": bars_30m,
                },
            })

        except Exception as e:
            print(f"   [略過 {sid}] {e}")

        if idx % 25 == 0 or idx == len(liquid_stocks):
            print(f"   進度: {idx}/{len(liquid_stocks)} | 成功寫入: {len(output_stocks)} 檔 | FinMind 請求: {client.request_count}")

    payload = {
        "meta": {
            "updatedAt": now_tw.strftime("%Y-%m-%d %H:%M:%S CST"),
            "marketDate": latest_market_date,
            "twseDate": market_dates["twse"],
            "tpexDate": market_dates["tpex"],
            "stockCount": len(output_stocks),
            "hasBrokerData": False,
            "source": "TWSE/TPEx OpenAPI + FinMind Free + yfinance",
            "finmindRequests": client.request_count,
            "reusedDailyCache": can_reuse_daily,
        },
        "stocks": output_stocks,
    }

    with open(CONFIG["OUTPUT_FILE"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(CONFIG["OUTPUT_FILE"]) / 1024.0
    print("=" * 72)
    print(f"✅ 已產出 {CONFIG['OUTPUT_FILE']} ({size_kb:.1f} KB, 共 {len(output_stocks)} 檔真實台股)")
    print("=" * 72)


if __name__ == "__main__":
    main()
