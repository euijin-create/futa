from datetime import date, timedelta
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template_string, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "oil_prices.csv"
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
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in OIL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0.0, col] = np.nan

    df["all"] = df[OIL_COLUMNS].mean(axis=1, skipna=True)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


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
        band_color = "badge-soft"
    elif distance < 1000:
        band = "보통"
        band_color = "badge-soft"
    else:
        band = "먼 편"
        band_color = "badge-soft"
    return info, distance, band, band_color


def summarize_period(df: pd.DataFrame, column: str, start_date: date, end_date: date):
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    period = df.loc[mask, ["date", column]].dropna(subset=[column])
    if period.empty:
        return None, period
    return float(period[column].mean()), period


def format_price(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def format_change(value):
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def decision_from_change(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return {
            "label": "판단 보류",
            "tone": "neutral",
            "headline": "데이터가 부족해서 지금은 판단을 보류합니다.",
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
      font-size: 1.9rem;
      font-weight: 900;
      line-height: 1;
      color: var(--ink);
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
      font-size: 1.85rem;
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
    .badge-soft {
      background: rgba(30, 95, 116, 0.1);
      color: var(--accent);
      border: 1px solid rgba(30, 95, 116, 0.18);
    }
    .form-label {
      font-weight: 800;
    }
    .small-help {
      color: var(--ink-soft);
      font-size: 0.92rem;
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
            항공권 가격을 예측하지 않고, 오피넷 국제 원유가격 추이를 바탕으로
            일본행 항공권을 지금 살지, 조금 더 기다릴지 아주 단순하게 참고하는 도구입니다.
          </p>
        </div>
        <div class="col-lg-4">
          <div class="bg-white text-dark rounded-4 p-3 p-lg-4 shadow-sm">
            <div class="fw-bold mb-2">아주 간단한 사용법</div>
            <div class="small text-secondary mb-1">1. 일본 목적지 선택</div>
            <div class="small text-secondary mb-1">2. 기준일 1개 선택</div>
            <div class="small text-secondary">3. 비교 기간만 고르면 끝</div>
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
          <div class="small-help mt-2">거리비례 구간제 관점으로 목적지 정보를 함께 보여줍니다.</div>
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
            <div class="fw-bold mb-1">기간은 보통 30일 추천</div>
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
            <div style="font-size: 1.7rem; font-weight: 900;">{{ previous_avg_display }}달러</div>
          </div>
          <div class="col-md-4">
            <div class="fw-bold">최근 기간 평균</div>
            <div style="font-size: 1.7rem; font-weight: 900;">{{ recent_avg_display }}달러</div>
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
            <div class="kpi-label">이전 기간 평균</div>
            <div class="kpi-value">{{ previous_avg_display }}</div>
            <div class="kpi-sub">달러</div>
          </div>
        </div>
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">최근 기간 평균</div>
            <div class="kpi-value">{{ recent_avg_display }}</div>
            <div class="kpi-sub">달러</div>
          </div>
        </div>
      </div>
    </section>

    <section class="mb-4">
      <div class="section-title">짧은 설명</div>
      <div class="hint-box">
        <div class="fw-bold mb-1">이전 기간 평균 원유가격은 {{ previous_avg_display }}달러, 최근 기간 평균 원유가격은 {{ recent_avg_display }}달러입니다.</div>
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
    df = load_data()
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
    recent_start = recent_end - timedelta(days=selected_window_days - 1)
    previous_end = recent_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=selected_window_days - 1)

    selected_col = "all"
    previous_avg, previous_period = summarize_period(df, selected_col, previous_start, previous_end)
    recent_avg, recent_period = summarize_period(df, selected_col, recent_start, recent_end)

    no_data_warning = None
    if previous_period.empty or recent_period.empty:
        no_data_warning = "해당 기간의 원유가격 데이터가 없습니다."

    diff = None
    percent_change = None
    if previous_avg is not None and recent_avg is not None:
        diff = recent_avg - previous_avg
        if previous_avg != 0:
            percent_change = diff / previous_avg * 100

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

    destination, distance_km, distance_band, _ = destination_info(selected_destination)
    destination_label = destination["label"]
    destination_sub = destination["sub"]

    line_df = df.melt(id_vars="date", value_vars=OIL_COLUMNS, var_name="원유", value_name="가격").dropna(subset=["가격"])
    line_df["원유"] = line_df["원유"].str.upper()
    line_fig = px.line(
        line_df,
        x="date",
        y="가격",
        color="원유",
        markers=True,
        title="날짜별 Dubai, Brent, WTI 원유가격 추이",
    )
    line_fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
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
        title=f"전체 평균 원유가격 비교",
        color_discrete_map={"이전 기간": "#8da0cb", "최근 기간": "#66c2a5"},
    )
    bar_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    bar_fig.update_layout(showlegend=False, yaxis_title="달러", margin=dict(l=10, r=10, t=60, b=10))
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
        previous_avg_display=format_price(previous_avg),
        recent_avg_display=format_price(recent_avg),
        percent_display=format_change(percent_change),
        destination_label=destination_label,
        destination_sub=destination_sub,
        distance_km=f"{distance_km:.0f}",
        distance_band=distance_band,
        change_sentence=change_sentence,
        short_takeaway=short_takeaway,
        line_chart=line_chart,
        bar_chart=bar_chart,
    )


if __name__ == "__main__":
    app.run(debug=True)

