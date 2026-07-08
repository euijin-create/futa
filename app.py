from datetime import date, timedelta
import calendar
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template_string, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OIL_DATA_PATH = BASE_DIR / "data" / "oil_prices.csv"
FX_DATA_PATH = BASE_DIR / "data" / "usd_krw.csv"

OIL_COLUMNS = ["dubai", "brent", "wti"]
SEOUL_COORDS = (37.5665, 126.9780)

JAPAN_DESTINATIONS = {
    "tokyo": {"label": "도쿄", "sub": "하네다 / 나리타", "lat": 35.6762, "lon": 139.6503},
    "osaka": {"label": "오사카", "sub": "간사이", "lat": 34.6937, "lon": 135.5023},
    "fukuoka": {"label": "후쿠오카", "sub": "후쿠오카", "lat": 33.5904, "lon": 130.4017},
    "sapporo": {"label": "삿포로", "sub": "신치토세", "lat": 43.0621, "lon": 141.3544},
    "okinawa": {"label": "오키나와", "sub": "나하", "lat": 26.2124, "lon": 127.6809},
    "nagoya": {"label": "나고야", "sub": "주부", "lat": 35.1815, "lon": 136.9066},
}


@lru_cache(maxsize=1)
def load_oil_data() -> pd.DataFrame:
    df = pd.read_csv(OIL_DATA_PATH)
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in OIL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0.0, col] = np.nan

    df["all"] = df[OIL_COLUMNS].mean(axis=1, skipna=True)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_fx_data() -> pd.DataFrame:
    df = pd.read_csv(FX_DATA_PATH)
    df.columns = [str(col).strip() for col in df.columns]

    date_col = next(
        (
            c
            for c in df.columns
            if c.lower() == "date" or "날짜" in c or "일자" in c
        ),
        None,
    )
    rate_col = next(
        (
            c
            for c in df.columns
            if "종가" in c or "달러" in c or "환율" in c or "usd" in c.lower()
        ),
        None,
    )

    if date_col is None or rate_col is None:
        if len(df.columns) >= 2:
            date_col = df.columns[0]
            rate_col = df.columns[1]
        else:
            raise ValueError("환율 CSV에서 날짜와 환율 컬럼을 찾을 수 없습니다.")

    fx = df[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "usd_krw"})
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx["usd_krw"] = (
        fx["usd_krw"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    fx["usd_krw"] = pd.to_numeric(fx["usd_krw"], errors="coerce")
    fx = fx.dropna(subset=["date", "usd_krw"]).sort_values("date").reset_index(drop=True)
    return fx


@lru_cache(maxsize=1)
def load_merged_data() -> pd.DataFrame:
    oil = load_oil_data()
    fx = load_fx_data()
    merged = pd.merge_asof(
        oil.sort_values("date"),
        fx.sort_values("date"),
        on="date",
        direction="backward",
    )

    merged["usd_krw"] = merged["usd_krw"].ffill().bfill()

    for col in OIL_COLUMNS + ["all"]:
        merged[f"{col}_krw"] = merged[col] * merged["usd_krw"]

    return merged


def safe_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError:
        return fallback


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return radius_km * c


def destination_info(code: str):
    info = JAPAN_DESTINATIONS[code]
    distance = haversine_km(SEOUL_COORDS[0], SEOUL_COORDS[1], info["lat"], info["lon"])
    if distance < 500:
        band = "가까운 편"
    elif distance < 1000:
        band = "보통"
    else:
        band = "먼 편"
    return info, distance, band


def summarize_period(df: pd.DataFrame, column: str, start_date: date, end_date: date):
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    period = df.loc[mask, ["date", column]].dropna(subset=[column])
    if period.empty:
        return None, period
    return float(period[column].mean()), period


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"₩{value:,.0f}"


def number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def decision_from_change(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return {
            "label": "판단 보류",
            "tone": "neutral",
            "headline": "지금은 판단을 보류해도 됩니다.",
            "text": "선택한 기간에 데이터가 없거나 이전 기간 평균이 0이라 증감률을 계산할 수 없습니다.",
        }
    if percent_change > 3:
        return {
            "label": "지금 발권 검토",
            "tone": "buy",
            "headline": "지금 사는 쪽이 유리해 보입니다.",
            "text": "최근 원유가격이 상승세라서, 원유가격 기준으로는 지금 발권을 검토하는 편이 좋습니다.",
        }
    if percent_change < -3:
        return {
            "label": "조금 더 대기",
            "tone": "wait",
            "headline": "조금 더 기다려도 괜찮아 보입니다.",
            "text": "최근 원유가격이 하락세라서, 원유가격 기준으로는 조금 더 기다려보는 선택지도 있습니다.",
        }
    return {
        "label": "차이 크지 않음",
        "tone": "neutral",
        "headline": "지금과 조금 기다리는 차이가 크지 않습니다.",
        "text": "최근 원유가격 변동이 크지 않아, 원유가격 기준으로는 발권 시점 차이가 크지 않아 보입니다.",
    }


def status_from_change(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return "neutral"
    if percent_change > 3:
        return "buy"
    if percent_change < -3:
        return "wait"
    return "neutral"


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    new_year, new_month_index = divmod(index, 12)
    return new_year, new_month_index + 1


def signal_for_date(df: pd.DataFrame, column: str, target_day: date, window_days: int):
    recent_start = target_day - timedelta(days=window_days - 1)
    previous_end = recent_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=window_days - 1)

    previous_avg, _ = summarize_period(df, column, previous_start, previous_end)
    recent_avg, _ = summarize_period(df, column, recent_start, target_day)

    if previous_avg is None or recent_avg is None or previous_avg == 0:
        return "neutral"

    percent_change = (recent_avg - previous_avg) / previous_avg * 100
    return status_from_change(percent_change)


def parse_calendar_month(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        year_str, month_str = value.split("-")
        year = int(year_str)
        month = int(month_str)
        return date(year, month, 1)
    except Exception:
        return fallback


def build_calendar_html(
    df: pd.DataFrame,
    column: str,
    view_month: date,
    window_days: int,
    selected_day: date,
    base_params: dict[str, str],
) -> str:
    month_calendar = calendar.Calendar(firstweekday=6)
    weekday_labels = ["일", "월", "화", "수", "목", "금", "토"]
    weeks = month_calendar.monthdatescalendar(view_month.year, view_month.month)

    prev_year, prev_month = shift_month(view_month.year, view_month.month, -1)
    next_year, next_month = shift_month(view_month.year, view_month.month, 1)

    prev_params = base_params | {"calendar_month": f"{prev_year}-{prev_month:02d}"}
    next_params = base_params | {"calendar_month": f"{next_year}-{next_month:02d}"}
    prev_query = "&".join(f"{k}={v}" for k, v in prev_params.items())
    next_query = "&".join(f"{k}={v}" for k, v in next_params.items())

    cells = []
    for week in weeks:
        week_cells = []
        for day in week:
            if day.month != view_month.month:
                week_cells.append('<div class="cal-cell empty"></div>')
                continue

            if day < df["date"].dt.date.min() or day > df["date"].dt.date.max():
                status = "neutral"
            else:
                status = signal_for_date(df, column, day, window_days)

            classes = f"cal-cell status-{status}"
            if day == selected_day:
                classes += " selected-day"
            if day == date.today():
                classes += " today-day"

            week_cells.append(
                f"""
                <div class="{classes}">
                  <div class="cal-bubble">{day.day}</div>
                </div>
                """
            )
        cells.append(f'<div class="cal-week">{"".join(week_cells)}</div>')

    weekday_html = "".join(f'<div class="cal-weekday">{label}</div>' for label in weekday_labels)

    return f"""
    <div class="calendar-wrap">
      <div class="calendar-head">
        <div>
          <div class="calendar-title">달력으로 보는 발권 신호</div>
          <div class="calendar-sub">초록 원은 "그날 샀으면 좋았던 날", 빨강 원은 "그날 샀으면 손해였던 날"입니다.</div>
        </div>
        <div class="calendar-nav">
          <a class="calendar-nav-btn" href="?{prev_query}">이전 달</a>
          <div class="calendar-badge">{view_month.strftime("%Y년 %m월")}</div>
          <a class="calendar-nav-btn" href="?{next_query}">다음 달</a>
        </div>
      </div>
      <div class="calendar-grid">
        {weekday_html}
        {"".join(cells)}
      </div>
      <div class="calendar-legend">
        <span><span class="legend-dot legend-buy"></span>사도 좋았음</span>
        <span><span class="legend-dot legend-wait"></span>사면 손해였음</span>
        <span><span class="legend-dot legend-neutral"></span>판단 보류</span>
        <span><span class="legend-dot legend-today"></span>기준일</span>
      </div>
    </div>
    """


HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>유타 - 일본행 발권 참고</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800;900&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --ink: #102542;
      --ink-soft: #4c5d73;
      --line: #dfe7f1;
      --panel: #f7f9fc;
      --green-bg: #e9f9ef;
      --green-line: #7bcf94;
      --red-bg: #ffecec;
      --red-line: #f39a9a;
      --neutral-bg: #eef3f8;
      --neutral-line: #c8d2df;
      --accent: #1e5f74;
    }
    body {
      font-family: "Noto Sans KR", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(121, 166, 191, 0.16), transparent 24%),
        radial-gradient(circle at left top, rgba(30, 95, 116, 0.12), transparent 20%),
        linear-gradient(180deg, #f8fbff 0%, #eef4fa 100%);
      color: var(--ink);
    }
    .hero {
      background: linear-gradient(135deg, #102542 0%, #1e5f74 52%, #79a6bf 100%);
      color: white;
      border-radius: 1.25rem;
      box-shadow: 0 16px 40px rgba(16, 37, 66, 0.16);
    }
    .hero h1 {
      font-size: clamp(1.8rem, 3vw, 2.7rem);
      font-weight: 900;
    }
    .hero p {
      line-height: 1.6;
      opacity: 0.96;
    }
    .panel {
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(223, 231, 241, 0.95);
      border-radius: 1rem;
      box-shadow: 0 10px 28px rgba(16, 37, 66, 0.06);
    }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1rem 1.1rem;
      height: 100%;
    }
    .kpi-label {
      color: var(--ink-soft);
      font-size: 0.92rem;
      margin-bottom: 0.35rem;
      font-weight: 700;
    }
    .kpi-value {
      font-size: 1.85rem;
      font-weight: 900;
      line-height: 1.1;
      color: var(--ink);
      word-break: keep-all;
    }
    .kpi-sub {
      margin-top: 0.35rem;
      color: #67788f;
      font-size: 0.9rem;
    }
    .decision-box {
      border-radius: 1.2rem;
      padding: 1.25rem 1.35rem;
      border-width: 2px;
      border-style: solid;
      margin-bottom: 1rem;
    }
    .decision-buy {
      background: var(--green-bg);
      border-color: var(--green-line);
      color: #14532d;
    }
    .decision-wait {
      background: var(--red-bg);
      border-color: var(--red-line);
      color: #7f1d1d;
    }
    .decision-neutral {
      background: var(--neutral-bg);
      border-color: var(--neutral-line);
      color: #334155;
    }
    .decision-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 0.35rem 0.8rem;
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      margin-bottom: 0.75rem;
      background: rgba(255,255,255,0.55);
    }
    .decision-title {
      font-size: 1.8rem;
      font-weight: 900;
      margin-bottom: 0.35rem;
    }
    .decision-text {
      font-size: 1.05rem;
      line-height: 1.6;
      margin-bottom: 0;
    }
    .hint-box {
      background: #fff8e7;
      border: 1px solid #f1d99f;
      border-radius: 1rem;
      padding: 1rem 1.1rem;
      color: #6c5200;
    }
    .chart-card {
      background: white;
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 0.4rem;
      box-shadow: 0 10px 28px rgba(16, 37, 66, 0.04);
    }
    .section-title {
      font-weight: 900;
      font-size: 1.1rem;
      margin-bottom: 0.75rem;
    }
    .mini-step {
      background: white;
      border: 1px solid var(--line);
      border-radius: 0.9rem;
      padding: 0.9rem 1rem;
      height: 100%;
    }
    .mini-step .step-num {
      width: 2rem;
      height: 2rem;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--accent);
      color: white;
      font-weight: 800;
      margin-bottom: 0.6rem;
    }
    .form-label {
      font-weight: 800;
    }
    .small-help {
      color: var(--ink-soft);
      font-size: 0.92rem;
    }
    .calendar-wrap {
      background: white;
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1rem;
      box-shadow: 0 10px 28px rgba(16, 37, 66, 0.04);
    }
    .calendar-head {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.75rem;
      flex-wrap: wrap;
    }
    .calendar-title {
      font-size: 1.05rem;
      font-weight: 900;
      color: var(--ink);
    }
    .calendar-sub {
      color: var(--ink-soft);
      font-size: 0.92rem;
      margin-top: 0.2rem;
    }
    .calendar-badge {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.4rem 0.85rem;
      font-weight: 800;
      color: var(--ink);
      white-space: nowrap;
    }
    .calendar-nav {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .calendar-nav-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.42rem 0.8rem;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      text-decoration: none;
      font-size: 0.88rem;
      font-weight: 800;
    }
    .calendar-nav-btn:hover {
      background: var(--panel);
      color: var(--ink);
    }
    .calendar-grid {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 0.45rem;
    }
    .cal-weekday {
      text-align: center;
      font-size: 0.82rem;
      font-weight: 800;
      color: var(--ink-soft);
      padding: 0.15rem 0;
    }
    .cal-cell {
      min-height: 5.8rem;
      border-radius: 0.85rem;
      border: 1px solid var(--line);
      background: #fbfdff;
      padding: 0.45rem;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .cal-cell.empty {
      background: transparent;
      border: 1px dashed rgba(223, 231, 241, 0.75);
    }
    .cal-bubble {
      width: 2.65rem;
      height: 2.65rem;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.95rem;
      font-weight: 900;
      color: white;
      box-shadow: 0 8px 16px rgba(16, 37, 66, 0.12);
    }
    .status-buy {
      background: rgba(74, 222, 128, 0.12);
      border-color: rgba(74, 222, 128, 0.45);
    }
    .status-buy .cal-bubble {
      background: #22c55e;
    }
    .status-wait {
      background: rgba(248, 113, 113, 0.12);
      border-color: rgba(248, 113, 113, 0.45);
    }
    .status-wait .cal-bubble {
      background: #ef4444;
    }
    .status-neutral {
      background: rgba(148, 163, 184, 0.12);
      border-color: rgba(148, 163, 184, 0.35);
    }
    .status-neutral .cal-bubble {
      background: #94a3b8;
    }
    .selected-day {
      box-shadow: inset 0 0 0 2px var(--accent);
    }
    .today-day::after {
      content: "오늘";
      position: absolute;
      right: 0.45rem;
      top: 0.35rem;
      font-size: 0.68rem;
      font-weight: 900;
      color: var(--accent);
    }
    .calendar-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem 1.2rem;
      margin-top: 0.85rem;
      color: var(--ink-soft);
      font-size: 0.9rem;
    }
    .calendar-legend span {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    .legend-dot {
      width: 0.7rem;
      height: 0.7rem;
      border-radius: 999px;
      display: inline-block;
    }
    .legend-buy {
      background: #22c55e;
    }
    .legend-wait {
      background: #ef4444;
    }
    .legend-neutral {
      background: #94a3b8;
    }
    .legend-today {
      background: var(--accent);
    }
  </style>
</head>
<body>
  <main class="container py-4 py-lg-5">
    <section class="hero p-4 p-lg-5 mb-4">
      <div class="row align-items-center g-3">
        <div class="col-lg-8">
          <div class="text-uppercase text-white-50 fw-bold mb-2" style="letter-spacing: .04em;">YUTA MVP</div>
          <h1 class="mb-3">원유가격 기반 일본행 발권 타이밍 참고 서비스</h1>
          <p class="mb-0">
            항공권 가격을 예측하지 않고, 오피넷 국제 원유가격 추이와 환율을 적용한 원화 기준으로
            일본행 항공권을 지금 살지, 조금 더 기다릴지 아주 단순하게 참고하는 도구입니다.
          </p>
        </div>
        <div class="col-lg-4">
          <div class="bg-white text-dark rounded-4 p-3 p-lg-4 shadow-sm">
            <div class="fw-bold mb-2">아주 간단한 사용법</div>
            <div class="small text-secondary mb-1">1. 일본 목적지 선택</div>
            <div class="small text-secondary mb-1">2. 기준일 1개만 선택</div>
            <div class="small text-secondary">3. 최근 기간은 자동 비교</div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel p-4 p-lg-4 mb-4">
      <form method="get" class="row g-3 align-items-end">
        <div class="col-lg-4 col-md-6">
          <label class="form-label" for="destination">일본 목적지</label>
          <select class="form-select" id="destination" name="destination">
            {% for code, item in destinations.items() %}
            <option value="{{ code }}" {% if code == selected_destination %}selected{% endif %}>
              {{ item.label }} ({{ item.sub }})
            </option>
            {% endfor %}
          </select>
          <div class="small-help mt-2">가고 싶은 도시만 고르면 됩니다.</div>
        </div>
        <div class="col-lg-3 col-md-6">
          <label class="form-label" for="as_of">기준일</label>
          <input class="form-control" type="date" id="as_of" name="as_of" value="{{ as_of }}">
          <div class="small-help mt-2">이 날짜를 최근 기간의 마지막 날로 사용합니다.</div>
        </div>
        <div class="col-lg-3 col-md-6">
          <label class="form-label" for="window_days">비교 기간</label>
          <select class="form-select" id="window_days" name="window_days">
            {% for days in window_options %}
            <option value="{{ days }}" {% if days == selected_window_days %}selected{% endif %}>최근 {{ days }}일</option>
            {% endfor %}
          </select>
          <div class="small-help mt-2">최근 기간과 그 이전 같은 기간을 자동 비교합니다.</div>
        </div>
        <div class="col-lg-2 col-md-6 d-grid">
          <button type="submit" class="btn btn-primary btn-lg" style="background: var(--accent); border-color: var(--accent);">결과 보기</button>
        </div>
      </form>
    </section>

    <section class="mb-4">
      {{ calendar_html|safe }}
    </section>

    <section class="mb-4">
      <div class="section-title">날짜 고르는 법</div>
      <div class="row g-3">
        <div class="col-md-4">
          <div class="mini-step">
            <div class="step-num">1</div>
            <div class="fw-bold mb-1">목적지 하나만 고르기</div>
            <div class="text-secondary">도쿄, 오사카, 후쿠오카처럼 가고 싶은 일본 도시를 선택하면 됩니다.</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="mini-step">
            <div class="step-num">2</div>
            <div class="fw-bold mb-1">기준일은 한 날짜만</div>
            <div class="text-secondary">예: 2026-07-08을 고르면 최근 30일과 그 전 30일을 자동으로 비교합니다.</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="mini-step">
            <div class="step-num">3</div>
            <div class="fw-bold mb-1">처음엔 최근 30일 추천</div>
            <div class="text-secondary">처음엔 최근 30일로 보면 가장 이해하기 쉽습니다. 더 짧게도 바꿀 수 있습니다.</div>
          </div>
        </div>
      </div>
    </section>

    {% if no_data_warning %}
    <div class="alert alert-warning border-0 shadow-sm">{{ no_data_warning }}</div>
    {% endif %}

    <section class="mb-3">
      <div class="decision-box decision-{{ decision.tone }}">
        <div class="decision-pill">{{ decision.label }}</div>
        <div class="decision-title">{{ decision.headline }}</div>
        <p class="decision-text mb-3">{{ decision.text }}</p>
        <div class="row g-3 align-items-center">
          <div class="col-md-4">
            <div class="fw-bold">증감률</div>
            <div style="font-size: 2.2rem; font-weight: 900;">{{ percent_display }}</div>
          </div>
          <div class="col-md-4">
            <div class="fw-bold">이전 기간 평균</div>
            <div style="font-size: 1.7rem; font-weight: 900;">{{ previous_avg_display }}</div>
          </div>
          <div class="col-md-4">
            <div class="fw-bold">최근 기간 평균</div>
            <div style="font-size: 1.7rem; font-weight: 900;">{{ recent_avg_display }}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="mb-4">
      <div class="row g-3">
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">목적지</div>
            <div class="kpi-value" style="font-size: 1.6rem;">{{ destination_label }}</div>
            <div class="kpi-sub">{{ destination_sub }}</div>
          </div>
        </div>
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">서울 기준 거리</div>
            <div class="kpi-value" style="font-size: 1.6rem;">{{ distance_km }}km</div>
            <div class="kpi-sub">{{ distance_band }} 기준 참고</div>
          </div>
        </div>
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">최근 환율</div>
            <div class="kpi-value" style="font-size: 1.6rem;">{{ fx_rate }}</div>
            <div class="kpi-sub">1달러 = 원화</div>
          </div>
        </div>
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">기준 화폐</div>
            <div class="kpi-value" style="font-size: 1.6rem;">원화 표시</div>
            <div class="kpi-sub">환율 적용 완료</div>
          </div>
        </div>
      </div>
    </section>

    <section class="mb-4">
      <div class="section-title">짧은 설명</div>
      <div class="hint-box">
        <div class="fw-bold mb-1">이전 기간 평균 원유가격은 {{ previous_avg_display }}, 최근 기간 평균 원유가격은 {{ recent_avg_display }}입니다.</div>
        <div class="mb-1">최근 기간은 이전 기간 대비 {{ change_sentence }}.</div>
        <div>따라서 원유가격 기준으로는 {{ short_takeaway }}로 참고할 수 있습니다.</div>
      </div>
    </section>

    <section class="mb-4">
      <div class="section-title">시각화</div>
      <div class="chart-card mb-3">
        {{ line_chart|safe }}
      </div>
      <div class="chart-card">
        {{ bar_chart|safe }}
      </div>
    </section>

    <section class="panel p-4">
      <div class="fw-bold mb-2">주의</div>
      <ul class="mb-0">
        <li>이 앱은 실제 항공권 가격 예측 서비스가 아닙니다.</li>
        <li>이 앱은 실제 항공사 유류할증료 고시표를 사용하지 않습니다.</li>
        <li>환율, 항공사 정책, 거리 구간, 발권일 기준 고시금액은 반영하지 않습니다.</li>
        <li>원유가격은 유류할증료의 간접 참고지표일 뿐입니다.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


@app.route("/")
def index():
    try:
        df = load_merged_data()
    except Exception as exc:
        return f"데이터를 불러오는 중 오류가 발생했습니다: {exc}", 500

    if df.empty:
        return "원유가격 데이터가 비어 있습니다.", 500

    min_date = df["date"].dt.date.min()
    max_date = df["date"].dt.date.max()
    default_as_of = max_date
    default_window_days = 30

    selected_destination = request.args.get("destination", "tokyo")
    if selected_destination not in JAPAN_DESTINATIONS:
        selected_destination = "tokyo"

    as_of = safe_date(request.args.get("as_of"), default_as_of)
    try:
        selected_window_days = int(request.args.get("window_days", default_window_days))
    except ValueError:
        selected_window_days = default_window_days
    if selected_window_days not in {7, 14, 30, 60}:
        selected_window_days = default_window_days

    recent_end = min(as_of, max_date)
    recent_start = max(min_date, recent_end - timedelta(days=selected_window_days - 1))
    previous_end = max(min_date, recent_start - timedelta(days=1))
    previous_start = max(min_date, previous_end - timedelta(days=selected_window_days - 1))

    calendar_month = parse_calendar_month(
        request.args.get("calendar_month"),
        date(recent_end.year, recent_end.month, 1),
    )

    previous_avg, previous_period = summarize_period(df, "all_krw", previous_start, previous_end)
    recent_avg, recent_period = summarize_period(df, "all_krw", recent_start, recent_end)

    no_data_warning = None
    if previous_period.empty or recent_period.empty:
        no_data_warning = "해당 기간의 원유가격 데이터가 없습니다."

    percent_change = None
    if previous_avg is not None and recent_avg is not None and previous_avg != 0:
        percent_change = (recent_avg - previous_avg) / previous_avg * 100

    decision = decision_from_change(percent_change)

    if percent_change is None:
        change_sentence = "증감률을 계산할 수 없습니다"
        short_takeaway = "판단 보류"
    elif percent_change > 0:
        change_sentence = f"{abs(percent_change):.2f}% 상승했습니다"
        short_takeaway = "지금 발권을 검토하는 쪽"
    elif percent_change < 0:
        change_sentence = f"{abs(percent_change):.2f}% 하락했습니다"
        short_takeaway = "조금 더 기다려보는 쪽"
    else:
        change_sentence = "거의 변동이 없습니다"
        short_takeaway = "발권 시점 차이가 크지 않은 쪽"

    destination, distance_km, distance_band = destination_info(selected_destination)

    fx_rate = df["usd_krw"].dropna().iloc[-1] if not df["usd_krw"].dropna().empty else None
    base_params = {
        "destination": selected_destination,
        "as_of": as_of.isoformat(),
        "window_days": str(selected_window_days),
    }
    calendar_html = build_calendar_html(
        df=df,
        column="all_krw",
        view_month=calendar_month,
        window_days=selected_window_days,
        selected_day=recent_end,
        base_params=base_params,
    )

    line_df = df.melt(
        id_vars=["date", "usd_krw"],
        value_vars=["dubai_krw", "brent_krw", "wti_krw"],
        var_name="원유",
        value_name="가격",
    ).dropna(subset=["가격"])
    line_df["원유"] = line_df["원유"].str.replace("_krw", "", regex=False).str.upper()

    line_fig = px.line(
        line_df,
        x="date",
        y="가격",
        color="원유",
        markers=True,
        title="날짜별 Dubai, Brent, WTI 원유가격 추이(원화 기준)",
    )
    line_fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    line_fig.update_yaxes(tickprefix="₩", separatethousands=True)
    line_chart = line_fig.to_html(full_html=False, include_plotlyjs="cdn")

    bar_values = pd.DataFrame(
        {
            "기간": ["이전 기간", "최근 기간"],
            "평균 가격": [previous_avg, recent_avg],
        }
    )
    bar_fig = px.bar(
        bar_values,
        x="기간",
        y="평균 가격",
        text="평균 가격",
        color="기간",
        title="전체 평균 원유가격 비교(원화 기준)",
        color_discrete_map={"이전 기간": "#8da0cb", "최근 기간": "#66c2a5"},
    )
    bar_fig.update_traces(texttemplate="₩%{text:,.0f}", textposition="outside")
    bar_fig.update_layout(showlegend=False, yaxis_title="원", margin=dict(l=10, r=10, t=60, b=10))
    bar_fig.update_yaxes(tickprefix="₩", separatethousands=True)
    bar_chart = bar_fig.to_html(full_html=False, include_plotlyjs=False)

    return render_template_string(
        HTML_TEMPLATE,
        destinations=JAPAN_DESTINATIONS,
        selected_destination=selected_destination,
        as_of=as_of.isoformat(),
        window_options=[7, 14, 30, 60],
        selected_window_days=selected_window_days,
        no_data_warning=no_data_warning,
        decision=decision,
        previous_avg_display=money(previous_avg),
        recent_avg_display=money(recent_avg),
        percent_display=pct(percent_change),
        destination_label=destination["label"],
        destination_sub=destination["sub"],
        distance_km=number(distance_km),
        distance_band=distance_band,
        fx_rate=money(fx_rate),
        change_sentence=change_sentence,
        short_takeaway=short_takeaway,
        calendar_html=calendar_html,
        line_chart=line_chart,
        bar_chart=bar_chart,
    )


if __name__ == "__main__":
    app.run(debug=True)
