#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXO Dealer Gamma Exposure (GEX) 模型  ─ LINE 推播版 v3
=======================================================
v6 新增 (籌碼因子層):
  [12] 三大法人期貨淨部位 (外資 TX 淨倉口數)
  [13] 小台散戶多空比 (全市場 MTX OI 扣除法人 → 散戶情緒,反指標候選)
  [14] TXO Put/Call Ratio (未平倉比)
  [15] factor_history.csv → 因子滾動 z-score (60日窗,≥20樣本才啟用)
  [16] LINE 摘要新增「📈 籌碼面」區塊 + 實驗性籌碼傾向分數
  [17] --backtest 新增因子 IC 分析 (Spearman 相關性 vs 隔日報酬)
  [18] safe_read_csv: 空檔/壞檔容錯,不再 EmptyDataError
  注意:籌碼訊號為實驗性質,方向解讀需累積樣本回測驗證後才可信

v5 (正確性與強健性):
  [7] 全面改用「資料實際交易日」: 圖標題/history/快照/回填全部以期交所資料日為準
  [8] 過期資料守門員: 資料日 ≤ 已記錄最後一日 → 跳過記錄/快照,LINE 提示未更新
  [9] 時區固定 Asia/Taipei (GitHub Actions 跑在 UTC,避免日期錯位)
  [10] 失敗 LINE 通知: 任何階段出錯立即推播原因,不再無聲死亡
  [11] 結算日偵測: 當天有合約到期 → 摘要加註結算釘盤提醒

v4 (邁向可驗證的量化系統):
  [4] 隔日報酬自動回填    → 每天抓 TX OHLC,回填昨日記錄的 next_open/high/low/close/ret
  [5] --backtest          → 回測引擎:按 regime 分組統計勝率/期望報酬/波動,含成本估算
  [6] 波動率自適應門檻    → 臨界判定改用近月 ATM IV 換算的日波動,取代固定 0.5%

v3 功能:
  [1] generate_signal()  → 量化訊號層  [2] gex_history.csv → 每日記錄
  [3] ΔOI 權重修正 (oi_snapshot.csv)

GitHub Actions 流程:
  1. python txo_gex.py        → 產生 txo_gex.png + summary.json + 更新 csv
  2. git commit & push         (png + csv)
  3. python txo_gex.py --push → 推 LINE

所需 GitHub Secrets: LINE_CHANNEL_ID / LINE_CHANNEL_SECRET / LINE_USER_ID
"""

import argparse
import io
import json
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

def tw_today():
    """台北時區的今天 (Actions 跑在 UTC,不能用 date.today())"""
    return datetime.now(ZoneInfo("Asia/Taipei")).date()

def norm_date(s):
    """期交所日期字串 → date (支援 YYYYMMDD / YYYY/MM/DD / YYYY-MM-DD)"""
    d = str(s).strip().replace("-", "").replace("/", "")
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    except Exception:
        return None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.stats import norm

MULT       = 50.0
R          = 0.017
HEADERS    = {"User-Agent": "Mozilla/5.0"}
GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "Andy0803/Investment-link-to-Line/main/txo_gex.png"
)
COST_PTS      = 12.0   # 單邊交易成本估計(大台,點):手續費+期交稅+滑價,回測用
SNAPSHOT_FILE = "oi_snapshot.csv"
FACTOR_FILE   = "factor_history.csv"
HISTORY_FILE  = "gex_history.csv"

# ══════════════════════════════════════════ utilities
def safe_read_csv(path):
    """[18] 空檔案/壞檔案 → 回傳空 DataFrame,不噴錯"""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError, pd.errors.ParserError):
        return pd.DataFrame()

def pick_col(cols, *keys):
    for c in cols:
        cc = str(c).replace(" ", "").lower()
        for k in keys:
            if k.lower() in cc:
                return c
    return None

def nth_wednesday(y, m, n):
    first  = date(y, m, 1)
    offset = (2 - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))

def parse_expiry(code):
    code = str(code).strip().upper()
    try:
        if "W" in code:
            ym, w = code.split("W")
            return nth_wednesday(int(ym[:4]), int(ym[4:6]), int(w))
        return nth_wednesday(int(code[:4]), int(code[4:6]), 3)
    except Exception:
        return None

# ══════════════════════════════════════════ Black-76
def black_price(F, K, T, sigma, cp):
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sq
    d2 = d1 - sq
    df = np.exp(-R * T)
    if cp == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol(price, F, K, T, cp):
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    if price <= intrinsic + 0.05:
        return np.nan
    try:
        return brentq(lambda s: black_price(F, K, T, s, cp) - price,
                      1e-3, 3.0, xtol=1e-6)
    except Exception:
        return np.nan

def gamma_grid(S, K, T, sigma):
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / sq
    return np.exp(-R * T) * norm.pdf(d1) / (S * sq)

# ══════════════════════════════════════════ data fetch
def fetch_openapi(path):
    r = requests.get(f"https://openapi.taifex.com.tw/v1/{path}",
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def fetch_txo_openapi():
    df = fetch_openapi("DailyMarketReportOpt")
    c_contract = pick_col(df.columns, "契約", "contract")
    c_month    = pick_col(df.columns, "到期", "month")
    c_strike   = pick_col(df.columns, "履約", "strike")
    c_cp       = pick_col(df.columns, "買賣權", "callput", "c/p")
    c_settle   = pick_col(df.columns, "結算", "settle")
    c_oi       = pick_col(df.columns, "未沖銷", "openinterest", "oi")
    c_sess     = pick_col(df.columns, "交易時段", "session")
    df = df[df[c_contract].astype(str).str.strip() == "TXO"].copy()
    if c_sess is not None:
        df = df[df[c_sess].astype(str).str.contains("一般|Regular", na=False)]
    return pd.DataFrame({
        "expiry_code": df[c_month].astype(str).str.strip(),
        "strike":      pd.to_numeric(df[c_strike], errors="coerce"),
        "cp":          df[c_cp].astype(str).str.strip().str[0].str.upper()
                       .map({"買": "C", "C": "C", "賣": "P", "P": "P"}),
        "settle":      pd.to_numeric(df[c_settle], errors="coerce"),
        "oi":          pd.to_numeric(df[c_oi], errors="coerce"),
    }).dropna(subset=["strike", "cp", "oi"])

_FUT_CACHE = {}
def fetch_fut_report():
    if "df" not in _FUT_CACHE:
        _FUT_CACHE["df"] = fetch_openapi("DailyMarketReportFut")
    return _FUT_CACHE["df"]

def fetch_tx_daily():
    """TX 近月完整日資料 (open/high/low/close/settle) → spot + 隔日回填用"""
    df = fetch_fut_report()
    c_contract = pick_col(df.columns, "契約", "contract")
    c_month    = pick_col(df.columns, "到期", "month")
    c_settle   = pick_col(df.columns, "結算", "settle")
    c_last     = pick_col(df.columns, "收盤", "last", "close")
    c_open     = pick_col(df.columns, "開盤", "open")
    c_high     = pick_col(df.columns, "最高", "high")
    c_low      = pick_col(df.columns, "最低", "low")
    df = df[df[c_contract].astype(str).str.strip() == "TX"].copy()
    df["m"] = df[c_month].astype(str).str.strip()
    df = df[df["m"].str.fullmatch(r"\d{6}")].sort_values("m")
    row = df.iloc[0]
    def num(c):
        v = pd.to_numeric(row[c], errors="coerce") if c else np.nan
        return float(v) if np.isfinite(v) and v > 0 else np.nan
    settle = num(c_settle)
    close  = num(c_last)
    c_date = pick_col(df.columns, "日期", "date")
    return {"open": num(c_open), "high": num(c_high), "low": num(c_low),
            "close": close if np.isfinite(close) else settle,
            "settle": settle if np.isfinite(settle) else close,
            "date": norm_date(row[c_date]) if c_date else None}

def fetch_txo_web(qdate):
    url  = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    data = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO",
            "queryDate": qdate.strftime("%Y/%m/%d"),
            "MarketCode": "0", "commodity_idt": "TXO"}
    r = requests.post(url, data=data, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    df = max(tables, key=len)
    return pd.DataFrame({
        "expiry_code": df[pick_col(df.columns, "到期")].astype(str).str.strip(),
        "strike":      pd.to_numeric(df[pick_col(df.columns, "履約")], errors="coerce"),
        "cp":          df[pick_col(df.columns, "買賣權")].astype(str).str.strip().str[0]
                       .map({"買": "C", "賣": "P", "C": "C", "P": "P"}),
        "settle":      pd.to_numeric(df[pick_col(df.columns, "結算")], errors="coerce"),
        "oi":          pd.to_numeric(df[pick_col(df.columns, "未沖銷")], errors="coerce"),
    }).dropna(subset=["strike", "cp", "oi"])

# ══════════════════════════════════════════ [3] OI 變化量修正
def apply_oi_change_weight(df, data_date):
    """
    用昨日 OI 快照計算 ΔOI,調整每檔合約在 GEX 中的權重。
    邏輯(啟發式):
      OI 增加 → 新倉建立,散戶淨買假設在「新增部位」上最可靠 → 權重放大
      OI 減少 → 部位平倉中,gamma 影響力衰退         → 權重收斂
    weight = clip(1 + 0.5 * ΔOI/OI, 0.6, 1.4),無昨日資料時 weight=1
    """
    df["w"] = 1.0
    if not os.path.exists(SNAPSHOT_FILE):
        print("[ΔOI] 無昨日快照,本次權重=1 (明天開始生效)")
        return df
    try:
        prev = safe_read_csv(SNAPSHOT_FILE)
        if len(prev) == 0:
            print("[ΔOI] 快照為空,權重=1")
            return df
        if "data_date" in prev.columns and len(prev) and \
           str(prev["data_date"].iloc[0]) == str(data_date):
            print("[ΔOI] 快照與本次資料同日,權重=1 (防呆)")
            return df
        prev["strike"] = pd.to_numeric(prev["strike"], errors="coerce")
        merged = df.merge(
            prev[["expiry_code", "strike", "cp", "oi"]],
            on=["expiry_code", "strike", "cp"],
            how="left", suffixes=("", "_prev"))
        doi   = merged["oi"] - merged["oi_prev"].fillna(0)
        ratio = doi / merged["oi"].clip(lower=1)
        df["w"] = np.clip(1 + 0.5 * ratio, 0.6, 1.4).values
        n_up   = int((doi > 0).sum())
        n_down = int((doi < 0).sum())
        print(f"[ΔOI] 權重已套用: 增倉 {n_up} 檔 / 減倉 {n_down} 檔")
    except Exception as e:
        print(f"[ΔOI] 快照讀取失敗({e}),權重=1")
    return df

def save_oi_snapshot(df, data_date):
    out = df[["expiry_code", "strike", "cp", "oi"]].copy()
    out["data_date"] = str(data_date)
    out.to_csv(SNAPSHOT_FILE, index=False)
    print(f"[ΔOI] OI 快照已存 (資料日 {data_date}) → {SNAPSHOT_FILE}")

# ══════════════════════════════════════════ [12-15] 籌碼因子層
def try_openapi(paths):
    """依序嘗試多個候選端點 (期交所端點名稱可能調整,防禦性設計)"""
    for p in paths:
        try:
            df = fetch_openapi(p)
            if len(df):
                return df
        except Exception:
            continue
    return None

def fetch_chip_factors():
    """抓三大法人/散戶/PCR 籌碼因子。任何一項失敗回 None,不影響主流程。"""
    out = {"foreign_net": None, "retail_ratio": None, "pcr_oi": None}

    # [12] 三大法人期貨 (取外資 TX 淨未平倉口數)
    inst = try_openapi([
        "MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate",
        "MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsByDate",
        "MajorInstitutionalTradersFut",
    ])
    inst_mtx_long = inst_mtx_short = None
    if inst is not None:
        try:
            print(f"[chip] 法人端點欄位: {list(inst.columns)}")
            c_prod = pick_col(inst.columns, "商品名稱", "商品", "契約", "contract", "product")
            c_id   = pick_col(inst.columns, "身份別", "身分別", "身份", "身分", "identity", "trader")
            c_ln   = pick_col(inst.columns, "多方未平倉口數", "多方未平倉契約金額", "多方未平倉", "longoi", "openinterestlong")
            c_sn   = pick_col(inst.columns, "空方未平倉口數", "空方未平倉契約金額", "空方未平倉", "shortoi", "openinterestshort")
            missing = [n for n,c in [("商品",c_prod),("身份",c_id),("多方OI",c_ln),("空方OI",c_sn)] if c is None]
            if missing:
                print(f"[chip] 法人欄位對不上,缺: {missing} (請把上面欄位列貼給開發者)")
            else:
                d = inst.copy()
                d["_p"]  = d[c_prod].astype(str)
                d["_id"] = d[c_id].astype(str)
                d["_l"]  = pd.to_numeric(d[c_ln].astype(str).str.replace(",", ""), errors="coerce")
                d["_s"]  = pd.to_numeric(d[c_sn].astype(str).str.replace(",", ""), errors="coerce")
                tx = d[d["_p"].str.contains("臺股期貨|台股期貨", regex=True, na=False)]
                fr = tx[tx["_id"].str.contains("外資", na=False)]
                if len(fr):
                    out["foreign_net"] = float(fr["_l"].sum() - fr["_s"].sum())
                mtx = d[d["_p"].str.contains("小型臺指|小型台指", na=False)]
                if len(mtx):
                    inst_mtx_long  = float(mtx["_l"].sum())
                    inst_mtx_short = float(mtx["_s"].sum())
        except Exception as e:
            print(f"[chip] 法人資料解析失敗: {type(e).__name__}: {e}")

    # [13] 小台散戶多空比 = (散戶多單-散戶空單)/全市場OI
    try:
        fut = fetch_fut_report()
        c_contract = pick_col(fut.columns, "契約", "contract")
        c_oi       = pick_col(fut.columns, "未沖銷", "openinterest", "oi")
        c_month    = pick_col(fut.columns, "到期", "month")
        mtx = fut[fut[c_contract].astype(str).str.strip() == "MTX"].copy()
        mtx = mtx[mtx[c_month].astype(str).str.strip().str.fullmatch(r"\d{6}")]
        total_oi = pd.to_numeric(mtx[c_oi], errors="coerce").sum()
        if total_oi > 0 and inst_mtx_long is not None:
            r_long  = total_oi - inst_mtx_long
            r_short = total_oi - inst_mtx_short
            out["retail_ratio"] = round(float((r_long - r_short) / total_oi), 4)
    except Exception as e:
        print(f"[chip] 小台散戶比計算失敗: {e}")

    # [14] TXO Put/Call Ratio (未平倉比)
    pcr = try_openapi(["PutCallRatio", "PutCallRatioByDate"])
    if pcr is not None:
        try:
            c = pick_col(pcr.columns, "未平倉量比率", "oiratio", "putcalloiratio")
            v = pd.to_numeric(str(pcr.iloc[-1][c]).replace("%", ""), errors="coerce")
            if np.isfinite(v):
                out["pcr_oi"] = float(v / 100 if v > 10 else v)
        except Exception as e:
            print(f"[chip] PCR 解析失敗: {e}")

    got = [k for k, v in out.items() if v is not None]
    print(f"[chip] 取得因子: {got if got else '無 (端點全數失敗,僅跑GEX)'}")
    return out

def update_factor_history(data_date, factors):
    """[15] 因子存檔 (同日覆蓋) → 回傳完整歷史供 z-score 計算"""
    fh = safe_read_csv(FACTOR_FILE)
    row = {"date": str(data_date), **factors}
    if len(fh):
        fh = fh[fh["date"] != row["date"]]
        fh = pd.concat([fh, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    else:
        fh = pd.DataFrame([row])
    fh.to_csv(FACTOR_FILE, index=False)
    return fh

def factor_z(fh, col, window=60, min_n=20):
    """滾動 z-score。樣本 < min_n 回 None (避免早期亂訊號)"""
    if col not in fh.columns:
        return None
    s = pd.to_numeric(fh[col], errors="coerce").dropna()
    if len(s) < min_n:
        return None
    w = s.tail(window)
    sd = w.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return float((s.iloc[-1] - w.mean()) / sd)

def chip_signal_block(fh, factors):
    """[16] 組 LINE 摘要的籌碼面區塊。z 可用時給傾向,否則只列原始值。"""
    lines, tilt_parts = ["📈 籌碼面"], []
    def arrow(z):
        return "↑偏多" if z > 0.5 else ("↓偏空" if z < -0.5 else "→中性")

    if factors.get("foreign_net") is not None:
        z = factor_z(fh, "foreign_net")
        if z is not None:
            lines.append(f"外資期貨淨倉 {factors['foreign_net']:+,.0f}口 ({z:+.1f}σ {arrow(z)})")
            tilt_parts.append(z)                      # 外資偏多 → 偏多
        else:
            lines.append(f"外資期貨淨倉 {factors['foreign_net']:+,.0f}口 (z累積中)")
    if factors.get("retail_ratio") is not None:
        z = factor_z(fh, "retail_ratio")
        if z is not None:
            lines.append(f"小台散戶多空比 {factors['retail_ratio']:+.1%} ({z:+.1f}σ)")
            tilt_parts.append(-z)                     # 散戶極多 → 反指標偏空
        else:
            lines.append(f"小台散戶多空比 {factors['retail_ratio']:+.1%} (z累積中)")
    if factors.get("pcr_oi") is not None:
        lines.append(f"P/C Ratio(OI) {factors['pcr_oi']:.2f}")

    if tilt_parts:
        t = float(np.mean(tilt_parts))
        d = "偏多" if t > 0.3 else ("偏空" if t < -0.3 else "中性")
        lines.append(f"籌碼傾向(實驗): {d} ({t:+.1f})")
    return "\n".join(lines) if len(lines) > 1 else None

# ══════════════════════════════════════════ demo data
def demo_data(spot=44382.0):
    rng = np.random.default_rng(7)
    rows, today = [], tw_today()
    expiries = [("THISWEEK", today + timedelta(days=4)),
                ("MONTH",    today + timedelta(days=18)),
                ("NEXTMON",  today + timedelta(days=46))]
    for name, exp in expiries:
        T = (exp - today).days / 365
        for K in np.arange(41000, 48050, 100.0):
            mny = np.log(K / spot)
            iv  = 0.20 + 0.9*max(-mny,0) + 0.25*max(mny,0) + 0.04*abs(mny)/np.sqrt(T)
            base = 4000 * np.exp(-(mny / 0.035)**2)
            c_oi = base*(1+rng.random()) + 6000*np.exp(-((K-45000)/220)**2)
            p_oi = base*(1+rng.random()) + 7000*np.exp(-((K-43000)/220)**2)
            for cp, oi in (("C", c_oi), ("P", p_oi)):
                rows.append(dict(expiry_code=name, strike=K, cp=cp,
                                 settle=black_price(spot,K,T,iv,cp),
                                 oi=max(int(oi),0), _T=T, _iv=iv))
    return pd.DataFrame(rows), spot

# ══════════════════════════════════════════ model core
def build_model(df, spot, today=None):
    today = today or tw_today()
    if "_T" in df.columns:
        df["T"] = df["_T"]
    else:
        df["expiry"] = df["expiry_code"].map(parse_expiry)
        df = df.dropna(subset=["expiry"])
        df["T"] = df["expiry"].map(lambda d: max((d-today).days+0.5, 0.5)/365)
        df = df[df["T"] < 0.6]

    df = df[(df["oi"] > 0) &
            (df["strike"] > spot*0.75) &
            (df["strike"] < spot*1.25)].copy()

    if "_iv" in df.columns:
        df["iv"] = df["_iv"]
    else:
        df["iv"] = [implied_vol(p,spot,k,t,cp)
                    for p,k,t,cp in zip(df["settle"],df["strike"],df["T"],df["cp"])]
        for code, g in df.groupby("expiry_code"):
            ok = g.dropna(subset=["iv"]).sort_values("strike")
            if len(ok) >= 3:
                fill = np.interp(g["strike"], ok["strike"], ok["iv"])
                df.loc[g.index,"iv"] = np.where(g["iv"].isna(), fill, g["iv"])
        df["iv"] = df["iv"].fillna(df["iv"].median()).clip(0.05, 2.0)

    df   = df.dropna(subset=["iv"])
    sign = np.where(df["cp"]=="C", 1.0, -1.0)
    # [3] 有效 OI = OI × ΔOI 權重 (無權重欄時 = 原始 OI)
    oi_eff = df["oi"].values * (df["w"].values if "w" in df.columns else 1.0)

    g_now     = gamma_grid(spot, df["strike"].values, df["T"].values, df["iv"].values)
    df["gex"] = sign * g_now * oi_eff * MULT * spot**2 * 0.01 / 1e8
    prof = df.groupby("strike")["gex"].agg(
        net="sum",
        call=lambda s: s[s>0].sum(),
        put=lambda s: s[s<0].sum()).reset_index()

    S_grid  = np.linspace(spot*0.94, spot*1.06, 241)
    K,T_,IV = df["strike"].values, df["T"].values, df["iv"].values
    G       = gamma_grid(S_grid[:,None], K[None,:], T_[None,:], IV[None,:])
    total_s = gaussian_filter1d(
        (G*(sign*oi_eff)[None,:]*MULT*S_grid[:,None]**2*0.01).sum(axis=1)/1e8, 4)

    near_T = df["T"].min()
    near   = df["T"].values <= near_T + 1e-9
    total_near_s = gaussian_filter1d(
        (G[:,near]*(sign[near]*oi_eff[near])[None,:]
         *MULT*S_grid[:,None]**2*0.01).sum(axis=1)/1e8, 4)

    def zero_cross(curve):
        idx = np.where(np.diff(np.sign(curve))!=0)[0]
        if not len(idx): return None
        xs = [S_grid[i]-curve[i]*(S_grid[i+1]-S_grid[i])/(curve[i+1]-curve[i]) for i in idx]
        return min(xs, key=lambda x: abs(x-spot))

    macro_zero = zero_cross(total_s)
    micro_flip = zero_cross(total_near_s)
    speed      = gaussian_filter1d(np.gradient(total_s, S_grid), 4)
    peak_S, valley_S = S_grid[np.argmax(speed)], S_grid[np.argmin(speed)]

    g_atk = gamma_grid(df["strike"].values, df["strike"].values*1.0001,
                       df["T"].values, df["iv"].values)
    df["wall_gex"] = sign*g_atk*oi_eff*MULT*df["strike"].values**2*0.01/1e8
    wallp  = df.groupby("strike")["wall_gex"].agg(
        call=lambda s: s[s>0].sum(),
        put=lambda s: s[s<0].sum()).reset_index()
    nearby    = wallp[(wallp["strike"]>spot*0.92)&(wallp["strike"]<spot*1.08)]
    call_wall = nearby.loc[nearby["call"].idxmax(),"strike"]
    put_wall  = nearby.loc[nearby["put"].idxmin(),"strike"]

    return dict(profile=prof, S_grid=S_grid, total=total_s, speed=speed,
                macro_zero=macro_zero, micro_flip=micro_flip,
                peak=float(peak_S), valley=float(valley_S),
                call_wall=float(call_wall), put_wall=float(put_wall),
                spot=float(spot))

# ══════════════════════════════════════════ [1] 量化訊號層
def generate_signal(m, atm_iv=None, expiry_today=False):
    """把 GEX 結構轉成今日判讀 + 上下關卡,回傳 (regime_code, critical, text)
    [6] 臨界門檻波動率自適應: threshold = spot × ATM_IV × sqrt(1/252) × 0.5
        (= 半個「一日標準差」),無 IV 時 fallback 固定 0.5%"""
    spot, mz = m["spot"], m["macro_zero"]

    if mz is None:
        return "UNKNOWN", False, "🧭 今日判讀\n翻轉點計算失敗,僅供參考"

    if atm_iv and np.isfinite(atm_iv):
        threshold = spot * atm_iv * np.sqrt(1 / 252) * 0.5
        thr_note  = f"(門檻 {threshold:.0f} 點 = 0.5×日波動, IV {atm_iv:.0%})"
    else:
        threshold = spot * 0.005
        thr_note  = f"(門檻 {threshold:.0f} 點 = 固定0.5%)"

    dist = spot - mz
    if dist > 0:
        regime_code = "POS_GAMMA"
        regime_txt  = "正Gamma區 → 盤整/釘盤傾向"
        bias        = "賣方策略有利,theta 收租順風;突破交易勝率低"
    else:
        regime_code = "NEG_GAMMA"
        regime_txt  = "負Gamma區 → 趨勢/波動放大"
        bias        = "順勢策略有利,買方(long gamma)順風;勿逆勢凹單"

    # 距離警示:波動率自適應臨界判定
    critical = abs(dist) < threshold
    if critical:
        regime_txt += " ⚠️臨界"
        bias = (f"距翻轉點僅 {abs(dist):.0f} 點 {thr_note},"
                f"結構隨時切換,留意 {mz:.0f} 攻防")

    # 上下最近關卡
    levels = {"Call Wall": m["call_wall"], "Put Wall": m["put_wall"],
              "Macro Zero": mz, "Speed Peak": m["peak"],
              "Speed Valley": m["valley"]}
    if m["micro_flip"]:
        levels["Micro Flip"] = m["micro_flip"]
    above = {k: v for k, v in levels.items() if v > spot}
    below = {k: v for k, v in levels.items() if v < spot}
    res = min(above.items(), key=lambda x: x[1]) if above else None
    sup = max(below.items(), key=lambda x: x[1]) if below else None

    lines = ["🧭 今日判讀", regime_txt,
             f"距 Macro Zero {dist:+.0f} 點"]
    if res:
        lines.append(f"上方關卡 {res[1]:.0f} ({res[0]})")
    if sup:
        lines.append(f"下方關卡 {sup[1]:.0f} ({sup[0]})")
    lines.append(f"💡 {bias}")
    if expiry_today:
        lines.append("📌 今日有合約結算,近月gamma極大,留意結算價釘盤效應")
    return regime_code, critical, "\n".join(lines)

# ══════════════════════════════════════════ [2] 回測資料記錄
NEXT_COLS = ["next_open", "next_high", "next_low", "next_close", "next_ret_pct"]

def append_history(m, regime_code, critical, data_date, factors=None):
    row = {
        "date":       str(data_date),
        "spot":       round(m["spot"]),
        "call_wall":  round(m["call_wall"]),
        "put_wall":   round(m["put_wall"]),
        "macro_zero": round(m["macro_zero"]) if m["macro_zero"] else None,
        "micro_flip": round(m["micro_flip"]) if m["micro_flip"] else None,
        "speed_peak": round(m["peak"]),
        "speed_valley": round(m["valley"]),
        "regime":     regime_code,
        "critical":   bool(critical),
    }
    for k, v in (factors or {}).items():
        row[k] = v
    for c in NEXT_COLS:
        row[c] = None
    hist = safe_read_csv(HISTORY_FILE)
    if len(hist):
        hist = hist[hist["date"] != row["date"]]          # 同日重跑 → 覆蓋
        new_row = pd.DataFrame([row])
        # 避免 all-NA 欄觸發 FutureWarning:對齊欄位後再接
        hist = pd.concat([hist, new_row], ignore_index=True) if len(hist) else new_row
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(HISTORY_FILE, index=False)
    print(f"[history] 已記錄 {row['date']} → {HISTORY_FILE} (共 {len(hist)} 筆)")

# ══════════════════════════════════════════ [4] 隔日報酬回填
def backfill_history(tx, data_date):
    """本次抓到的 TX OHLC(屬於 data_date 交易日) = 上一筆記錄的「隔日」走勢 → 回填。
    守門:只回填日期「早於 data_date」的記錄,杜絕 API 過期時自己填自己。"""
    if not os.path.exists(HISTORY_FILE) or tx is None or data_date is None:
        return
    hist = safe_read_csv(HISTORY_FILE)
    if len(hist) == 0:
        return
    for c in NEXT_COLS + ["critical"]:
        if c not in hist.columns:
            hist[c] = None
    cand = hist[(hist["date"] < str(data_date)) & (hist["next_close"].isna())]
    if len(cand) == 0:
        return
    idx      = cand.index[-1]                       # 最近一筆未回填 = 上一個交易日
    row_date = pd.to_datetime(hist.loc[idx, "date"]).date()
    if (data_date - row_date).days > 5:             # 中斷太久不硬填,避免錯置
        print(f"[backfill] {row_date} 與資料日 {data_date} 相距過遠,跳過")
        return
    spot_prev = float(hist.loc[idx, "spot"])
    hist.loc[idx, "next_open"]  = tx["open"]
    hist.loc[idx, "next_high"]  = tx["high"]
    hist.loc[idx, "next_low"]   = tx["low"]
    hist.loc[idx, "next_close"] = tx["close"]
    if np.isfinite(tx["close"]) and spot_prev > 0:
        hist.loc[idx, "next_ret_pct"] = round((tx["close"] - spot_prev) / spot_prev * 100, 3)
    hist.to_csv(HISTORY_FILE, index=False)
    print(f"[backfill] 已回填 {row_date} 的隔日走勢 (close {tx['close']:.0f})")

# ══════════════════════════════════════════ [5] 回測引擎
def run_backtest():
    """按 regime 分組統計隔日表現。用法: python txo_gex.py --backtest"""
    h = safe_read_csv(HISTORY_FILE)
    if len(h) == 0:
        print("尚無歷史資料"); return
    h = h.dropna(subset=["next_ret_pct"]).copy()
    if len(h) < 10:
        print(f"有效樣本僅 {len(h)} 筆 (<10),統計不具意義,先累積資料"); return

    h["next_range_pct"] = (h["next_high"] - h["next_low"]) / h["spot"] * 100
    cost_pct = COST_PTS * 2 / h["spot"].mean() * 100   # 來回成本(%)

    print(f"\n{'='*58}")
    print(f" GEX 回測報告   樣本 {len(h)} 個交易日   來回成本 {cost_pct:.3f}%")
    print(f"{'='*58}")
    groups = [("全部", h),
              ("正Gamma (POS)", h[h.regime == "POS_GAMMA"]),
              ("負Gamma (NEG)", h[h.regime == "NEG_GAMMA"]),
              ("臨界日",        h[h.critical == True]),
              ("非臨界日",      h[h.critical == False])]
    print(f"{'分組':<14}{'N':>4}{'勝率%':>8}{'均報酬%':>9}{'標準差%':>9}{'日振幅%':>9}")
    for name, g in groups:
        if len(g) == 0:
            continue
        win = (g.next_ret_pct > 0).mean() * 100
        print(f"{name:<14}{len(g):>4}{win:>8.1f}{g.next_ret_pct.mean():>9.3f}"
              f"{g.next_ret_pct.std():>9.3f}{g.next_range_pct.mean():>9.3f}")
    # 核心假設檢驗:負gamma日振幅應大於正gamma
    pos = h[h.regime == "POS_GAMMA"]["next_range_pct"]
    neg = h[h.regime == "NEG_GAMMA"]["next_range_pct"]
    if len(pos) >= 5 and len(neg) >= 5:
        print(f"\n[假設檢驗] 負Gamma日均振幅 {neg.mean():.2f}% vs 正Gamma {pos.mean():.2f}%"
              f" → {'✅ 符合模型預期' if neg.mean() > pos.mean() else '❌ 與假設相反,模型需檢討'}")
    # [17] 因子 IC:Spearman 相關 (因子值 vs 隔日報酬)
    from scipy.stats import spearmanr
    fac_cols = [c for c in ("foreign_net", "retail_ratio", "pcr_oi") if c in h.columns]
    rows = []
    for c in fac_cols:
        sub = h.dropna(subset=[c, "next_ret_pct"])
        if len(sub) >= 15:
            ic, p = spearmanr(sub[c], sub["next_ret_pct"])
            rows.append((c, len(sub), ic, p))
    if rows:
        print(f"\n[因子IC] Spearman(因子, 隔日報酬)  |IC|>0.1且p<0.1才值得關注")
        for c, n, ic, p in rows:
            print(f"  {c:<14} N={n:<4} IC={ic:+.3f}  p={p:.3f}")

    print(f"\n註:報酬為「訊號日結算→隔日結算」的被動持有,未含方向策略;")
    print(f"    任何策略均需扣 {cost_pct:.3f}% 來回成本後仍為正才有意義。")

# ══════════════════════════════════════════ plotting
def plot_model(m, out="txo_gex.png", title_date=None):
    p, spot    = m["profile"], m["spot"]
    title_date = title_date or tw_today().strftime("%Y/%m/%d")

    lo, hi = spot * 0.94, spot * 1.06
    p = p[(p["strike"] >= lo) & (p["strike"] <= hi)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))

    ax1.axvspan(spot*0.98, spot*1.02, color="yellow", alpha=0.18, label="ATM Zone (±2%)")
    w = (p["strike"].diff().median() or 100) * 0.6
    ax1.bar(p["strike"], p["call"], width=w, color="mediumseagreen", alpha=0.65, label="Call $GEX")
    ax1.bar(p["strike"], p["put"],  width=w, color="salmon",         alpha=0.75, label="Put $GEX")
    ax1.plot(p["strike"], p["net"], color="gray", lw=1.4, label="Net $GEX")
    if lo < m["call_wall"] < hi:
        ax1.axvline(m["call_wall"], color="green", lw=2.4, label=f"Call Wall: {m['call_wall']:.0f}")
    if lo < m["put_wall"] < hi:
        ax1.axvline(m["put_wall"],  color="red",   lw=2.4, label=f"Put Wall: {m['put_wall']:.0f}")
    if m["micro_flip"]:
        ax1.axvline(m["micro_flip"], color="orange", lw=2,  label=f"Micro Flip: {m['micro_flip']:.0f}")
    if m["macro_zero"]:
        ax1.axvline(m["macro_zero"], color="purple", ls="-.", lw=1.8,
                    label=f"Macro Zero: {m['macro_zero']:.0f}")
    ax1.axvline(spot, color="navy", ls="--", lw=1.8, label=f"Previous Price: {spot:.0f}")
    ax1.set_xlim(lo, hi)
    notes = []
    if not (lo < m["call_wall"] < hi):
        notes.append(f"Call Wall: {m['call_wall']:.0f} (out of view →)")
    if not (lo < m["put_wall"] < hi):
        notes.append(f"← Put Wall: {m['put_wall']:.0f} (out of view)")
    if notes:
        ax1.text(0.99, 0.02, "   ".join(notes), transform=ax1.transAxes,
                 ha="right", va="bottom", fontsize=9, color="dimgray")
    ax1.set_title(f"TXO Net Exposure & Pre-Market Key Levels ({title_date})", fontsize=14)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)

    ax2.plot(m["S_grid"], m["speed"], color="purple", lw=2, label="dGEX/dSpot (Hedging Speed)")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.axvline(m["peak"],   color="green", ls=":", lw=2, label=f"Peak: {m['peak']:.0f}")
    ax2.axvline(m["valley"], color="red",   ls=":", lw=2, label=f"Valley: {m['valley']:.0f}")
    ax2.axvline(spot, color="navy", ls="--", lw=1.8)
    ax2.set_xlim(lo, hi)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"[saved] {out}")

# ══════════════════════════════════════════ LINE push
def get_line_token():
    r = requests.post(
        "https://api.line.me/oauth2/v3/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials",
              "client_id":     os.environ["LINE_CHANNEL_ID"],
              "client_secret": os.environ["LINE_CHANNEL_SECRET"]},
        timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

def push_line(summary_text, with_image=True):
    uid = os.environ.get("LINE_USER_ID", "")
    if not uid or not os.environ.get("LINE_CHANNEL_ID"):
        print("[LINE] 環境變數未設定,跳過")
        return
    try:
        token = get_line_token()
        h = {"Authorization": f"Bearer {token}",
             "Content-Type": "application/json"}
        messages = [{"type": "text", "text": summary_text}]
        if with_image:
            image_url = f"{GITHUB_RAW}?t={int(time.time())}"
            messages.append({"type": "image",
                             "originalContentUrl": image_url,
                             "previewImageUrl":    image_url})
        body = {"to": uid, "messages": messages}
        r = requests.post("https://api.line.me/v2/bot/message/push",
                          headers=h, json=body, timeout=15)
        r.raise_for_status()
        print("[LINE] push OK")
    except Exception as e:
        print(f"[LINE] push 失敗: {e}")

def make_summary(m, signal_text, data_date):
    mz = f"{m['macro_zero']:.0f}" if m['macro_zero'] else "N/A"
    mf = f"{m['micro_flip']:.0f}" if m['micro_flip'] else "N/A"
    return (
        f"📊 TXO GEX  資料日 {data_date}\n"
        f"─────────────────\n"
        f"現價　　　 {m['spot']:.0f}\n"
        f"Call Wall　{m['call_wall']:.0f}\n"
        f"Put Wall　 {m['put_wall']:.0f}\n"
        f"Macro Zero {mz}\n"
        f"Micro Flip {mf}\n"
        f"Speed Peak {m['peak']:.0f}\n"
        f"Speed Val  {m['valley']:.0f}\n"
        f"─────────────────\n"
        f"{signal_text}"
    )

# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=float, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--backtest", action="store_true",
                    help="輸出回測統計報告")
    args = ap.parse_args()

    if args.backtest:
        run_backtest()
        return

    # ── 第二階段:只推 LINE
    if args.push:
        with open("summary.json") as f:
            d = json.load(f)
        push_line(d["text"], with_image=d.get("image", True))
        return

    # ── 第一階段:包例外,失敗時 LINE 通知 [10]
    try:
        run_pipeline(args)
    except Exception as e:
        msg = f"⚠️ TXO GEX 執行失敗\n{type(e).__name__}: {e}"
        print(msg)
        push_line(msg, with_image=False)
        raise                              # 讓 workflow 顯示紅色失敗

def run_pipeline(args):
    tx = None
    if args.demo:
        df, spot = demo_data(args.spot or 44382.0)
    else:
        try:
            df = fetch_txo_openapi()
            if len(df) == 0:
                raise ValueError("empty")
            print(f"[OpenAPI] TXO rows: {len(df)}")
        except Exception as e:
            print(f"[OpenAPI failed: {e}] → fallback")
            df = fetch_txo_web(tw_today())
        tx   = fetch_tx_daily()
        spot = args.spot or tx["settle"]
        print(f"[spot] {spot:.0f}")

    # [7] 一切以資料實際交易日為準
    data_date = (tx or {}).get("date") or tw_today()
    print(f"[data_date] {data_date}")

    # [8] 過期資料守門員
    if not args.demo:
        hist = safe_read_csv(HISTORY_FILE)
        if len(hist) and "date" in hist.columns and \
           str(data_date) <= str(hist["date"].iloc[-1]):
            note = (f"ℹ️ TXO GEX:期交所資料尚未更新\n"
                    f"最新資料日仍為 {hist['date'].iloc[-1]},稍後再試")
            print(note)
            with open("summary.json", "w") as f:
                json.dump({"text": note, "image": False}, f, ensure_ascii=False)
            return

    backfill_history(tx, data_date)              # [4]+[7] 回填上一筆的隔日走勢
    df = apply_oi_change_weight(df, data_date)   # [3]+[7] ΔOI 權重
    m  = build_model(df, spot)
    save_oi_snapshot(df, data_date)              # [3]+[7] 快照帶資料日

    # [6] 近月 ATM IV → 自適應臨界門檻
    atm_iv = None
    try:
        dd     = df.copy()
        dd     = dd[dd["T"] == dd["T"].min()] if "T" in dd.columns else dd
        atm_iv = float(dd.loc[(dd["strike"] - spot).abs().idxmin(), "iv"])
    except Exception:
        pass

    # [11] 結算日偵測:今天(台北)是否有合約到期
    expiry_today = False
    if not args.demo:
        try:
            expiries     = set(df["expiry_code"].map(parse_expiry).dropna())
            expiry_today = tw_today() in expiries
        except Exception:
            pass

    # [12-16] 籌碼因子 (任何失敗不影響 GEX 主流程)
    factors, chip_block = {}, None
    if not args.demo:
        try:
            factors    = fetch_chip_factors()
            fh         = update_factor_history(data_date, factors)
            chip_block = chip_signal_block(fh, factors)
        except Exception as e:
            print(f"[chip] 籌碼模組失敗(不影響GEX): {e}")

    regime_code, critical, signal_text = generate_signal(m, atm_iv, expiry_today)
    if chip_block:
        signal_text = signal_text + "\n─────────────────\n" + chip_block
    append_history(m, regime_code, critical, data_date,
                   {k: v for k, v in factors.items() if v is not None})

    summary = make_summary(m, signal_text, data_date)
    print("=" * 46)
    print(summary)
    print("=" * 46)

    plot_model(m, title_date=data_date.strftime("%Y/%m/%d"))
    with open("summary.json", "w") as f:
        json.dump({"text": summary, "image": True}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
