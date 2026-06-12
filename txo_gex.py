#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXO Dealer Gamma Exposure (GEX) 模型  ─ LINE 推播版 v3
=======================================================
v3 新增:
  [1] generate_signal()  → 量化訊號層:今日 gamma 區域判讀 + 上下關卡,直接寫進 LINE 摘要
  [2] gex_history.csv    → 每日關鍵價位自動記錄,累積回測資料
  [3] OI 變化量修正      → 用 ΔOI 調整各合約權重,精修 dealer 部位假設
                           (oi_snapshot.csv 存每日 OI,隔日計算變化量)

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
from datetime import date, timedelta

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
SNAPSHOT_FILE = "oi_snapshot.csv"
HISTORY_FILE  = "gex_history.csv"

# ══════════════════════════════════════════ utilities
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

def fetch_spot_openapi():
    df = fetch_openapi("DailyMarketReportFut")
    c_contract = pick_col(df.columns, "契約", "contract")
    c_month    = pick_col(df.columns, "到期", "month")
    c_settle   = pick_col(df.columns, "結算", "settle")
    c_last     = pick_col(df.columns, "收盤", "last", "close")
    df = df[df[c_contract].astype(str).str.strip() == "TX"].copy()
    df["m"] = df[c_month].astype(str).str.strip()
    df = df[df["m"].str.fullmatch(r"\d{6}")].sort_values("m")
    row = df.iloc[0]
    px  = pd.to_numeric(row[c_settle], errors="coerce")
    if not np.isfinite(px) or px <= 0:
        px = pd.to_numeric(row[c_last], errors="coerce")
    return float(px)

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
def apply_oi_change_weight(df):
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
        prev = pd.read_csv(SNAPSHOT_FILE)
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

def save_oi_snapshot(df):
    df[["expiry_code", "strike", "cp", "oi"]].to_csv(SNAPSHOT_FILE, index=False)
    print(f"[ΔOI] 今日 OI 快照已存 → {SNAPSHOT_FILE}")

# ══════════════════════════════════════════ demo data
def demo_data(spot=44382.0):
    rng = np.random.default_rng(7)
    rows, today = [], date.today()
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
    today = today or date.today()
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
def generate_signal(m):
    """把 GEX 結構轉成今日判讀 + 上下關卡,回傳 (regime_code, text)"""
    spot, mz = m["spot"], m["macro_zero"]

    if mz is None:
        return "UNKNOWN", "🧭 今日判讀\n翻轉點計算失敗,僅供參考"

    dist = spot - mz
    if dist > 0:
        regime_code = "POS_GAMMA"
        regime_txt  = "正Gamma區 → 盤整/釘盤傾向"
        bias        = "賣方策略有利,theta 收租順風;突破交易勝率低"
    else:
        regime_code = "NEG_GAMMA"
        regime_txt  = "負Gamma區 → 趨勢/波動放大"
        bias        = "順勢策略有利,買方(long gamma)順風;勿逆勢凹單"

    # 距離警示:離翻轉點 0.5% 以內視為臨界狀態
    if abs(dist) < spot * 0.005:
        regime_txt += " ⚠️臨界"
        bias = f"距翻轉點僅 {abs(dist):.0f} 點,結構隨時切換,留意 {mz:.0f} 攻防"

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
    return regime_code, "\n".join(lines)

# ══════════════════════════════════════════ [2] 回測資料記錄
def append_history(m, regime_code):
    row = {
        "date":       str(date.today()),
        "spot":       round(m["spot"]),
        "call_wall":  round(m["call_wall"]),
        "put_wall":   round(m["put_wall"]),
        "macro_zero": round(m["macro_zero"]) if m["macro_zero"] else None,
        "micro_flip": round(m["micro_flip"]) if m["micro_flip"] else None,
        "speed_peak": round(m["peak"]),
        "speed_valley": round(m["valley"]),
        "regime":     regime_code,
    }
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist = hist[hist["date"] != row["date"]]          # 同日重跑 → 覆蓋
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(HISTORY_FILE, index=False)
    print(f"[history] 已記錄 {row['date']} → {HISTORY_FILE} (共 {len(hist)} 筆)")

# ══════════════════════════════════════════ plotting
def plot_model(m, out="txo_gex.png", title_date=None):
    p, spot    = m["profile"], m["spot"]
    title_date = title_date or date.today().strftime("%Y/%m/%d")

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

def push_line(summary_text):
    uid = os.environ.get("LINE_USER_ID", "")
    if not uid or not os.environ.get("LINE_CHANNEL_ID"):
        print("[LINE] 環境變數未設定,跳過")
        return
    try:
        token     = get_line_token()
        image_url = f"{GITHUB_RAW}?t={int(time.time())}"
        h = {"Authorization": f"Bearer {token}",
             "Content-Type": "application/json"}
        body = {"to": uid, "messages": [
            {"type": "text", "text": summary_text},
            {"type": "image",
             "originalContentUrl": image_url,
             "previewImageUrl":    image_url}
        ]}
        r = requests.post("https://api.line.me/v2/bot/message/push",
                          headers=h, json=body, timeout=15)
        r.raise_for_status()
        print("[LINE] push OK")
    except Exception as e:
        print(f"[LINE] push 失敗: {e}")

def make_summary(m, signal_text):
    mz = f"{m['macro_zero']:.0f}" if m['macro_zero'] else "N/A"
    mf = f"{m['micro_flip']:.0f}" if m['micro_flip'] else "N/A"
    return (
        f"📊 TXO GEX  {date.today()}\n"
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
    args = ap.parse_args()

    # ── 第二階段:只推 LINE
    if args.push:
        with open("summary.json") as f:
            push_line(json.load(f)["text"])
        return

    # ── 第一階段:抓資料 → 模型 → 圖 + 訊號 + 記錄
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
            df = fetch_txo_web(date.today())
        spot = args.spot or fetch_spot_openapi()
        print(f"[spot] {spot:.0f}")

    df = apply_oi_change_weight(df)        # [3] ΔOI 權重
    m  = build_model(df, spot)
    save_oi_snapshot(df)                   # [3] 存今日快照供明日使用

    regime_code, signal_text = generate_signal(m)   # [1] 訊號層
    append_history(m, regime_code)                  # [2] 回測記錄

    summary = make_summary(m, signal_text)
    print("=" * 46)
    print(summary)
    print("=" * 46)

    plot_model(m)
    with open("summary.json", "w") as f:
        json.dump({"text": summary}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
