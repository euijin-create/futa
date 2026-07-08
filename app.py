from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template_string, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "oil_prices.csv"
OIL_COLUMNS = ["dubai", "brent", "wti"]
OIL_LABELS = {
    "dubai": "Dubai",
    "brent": "Brent",
    "wti": "WTI",
    "all": "전체 평균",
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
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def safe_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError:
        return fallback


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


def recommendation_text(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return "판단 보류"
    if percent_change > 3:
        return "최근 원유가격이 상승세입니다. 유류할증료가 오를 가능성을 고려하면, 원유가격 기준으로는 지금 발권을 검토하는 것이 유리해 보입니다."
    if percent_change < -3:
        return "최근 원유가격이 하락세입니다. 유류할증료가 내려갈 가능성을 고려하면, 조금 기다려보는 것도 선택지가 될 수 있습니다."
    return "최근 원유가격 변동이 크지 않습니다. 원유가격 기준으로는 발권 시점 차이가 크지 않아 보입니다."


def recommendation_tag(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return "판단 보류"
    if percent_change > 3:
        return "지금 발권 검토"
    if percent_change < -3:
        return "조금 더 대기"
    return "차이 크지 않음"


HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>유타 - 원유가격 기반 일본행 항공권 발권 타이밍 참고 서비스</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --ink: #102542;
      --ink-soft: #3b4c63;
      --line: #e6ebf3;
      --panel: #f7f9fc;
      --accent: #1e5f74;
      --accent-2: #79a6bf;
      --warm: #fff8e7;
      --warm-line: #f1d99f;
    }
    body {
      font-family: "Noto Sans KR", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(121, 166, 191, 0.18), transparent 26%),
        radial-gradient(circle at left top, rgba(30, 95, 116, 0.14), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, #eef4fa 100%);
      color: var(--ink);
    }
    .hero {
      border-radius: 1.25rem;
      background: linear-gradient(135deg, #102542 0%, #1e5f74 52%, #79a6bf 100%);
      color: white;
      box-shadow: 0 16px 40px rgba(16, 37, 66, 0.16);
    }
    .hero h1 {
      font-size: clamp(1.8rem, 3vw, 2.6rem);
      font-weight: 800;
    }
    .hero p {
      opacity: 0.96;
      line-height: 1.6;
      max-width: 64rem;
    }
    .panel {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(230, 235, 243, 0.95);
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
      margin-bottom: 0.4rem;
    }
    .kpi-value {
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
      color: var(--ink);
    }
    .kpi-sub {
      margin-top: 0.35rem;
      color: #68788f;
      font-size: 0.9rem;
    }
    .note-box {
      background: var(--warm);
      border: 1px solid var(--warm-line);
      border-radius: 1rem;
      padding: 1rem 1.1rem;
      color: #6c5200;
    }
    .form-label {
      font-weight: 700;
      color: var(--ink);
    }
    .muted-title {
      color: var(--ink-soft);
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.01em;
      text-transform: uppercase;
    }
    .chart-card {
      background: white;
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 0.5rem;
      box-shadow: 0 10px 28px rgba(16, 37, 66, 0.04);
    }
  </style>
</head>
<body>
  <main class="container py-4 py-lg-5">
    <section class="hero p-4 p-lg-5 mb-4">
      <div class="row align-items-center g-3">
        <div class="col-lg-8">
          <div class="muted-title text-white-50 mb-2">YUTA MVP</div>
          <h1 class="mb-3">원유가격 기반 일본행 항공권 발권 타이밍 참고 서비스</h1>
          <p class="mb-0">
            이 서비스는 항공권 가격이나 실제 유류할증료를 예측하지 않습니다.
            오피넷 국제 원유가격 추이를 바탕으로 발권 타이밍을 참고할 수 있게 돕는 초간단 MVP입니다.
          </p>
        </div>
        <div class="col-lg-4">
          <div class="bg-white text-dark rounded-4 p-3 p-lg-4 shadow-sm">
            <div class="fw-bold mb-2">기준</div>
            <div class="small text-secondary">원유가격은 유류할증료의 간접 참고지표입니다.</div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel p-4 p-lg-4 mb-4">
      <form method="get" class="row g-3 align-items-end">
        <div class="col-lg-3 col-md-6">
          <label class="form-label" for="oil">원유 종류</label>
          <select class="form-select" id="oil" name="oil">
            {% for value, label in oil_options.items() %}
            <option value="{{ value }}" {% if value == selected_oil %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-lg-2 col-md-6">
          <label class="form-label" for="previous_start">이전 시작</label>
          <input class="form-control" type="date" id="previous_start" name="previous_start" value="{{ previous_start }}">
        </div>
        <div class="col-lg-2 col-md-6">
          <label class="form-label" for="previous_end">이전 종료</label>
          <input class="form-control" type="date" id="previous_end" name="previous_end" value="{{ previous_end }}">
        </div>
        <div class="col-lg-2 col-md-6">
          <label class="form-label" for="recent_start">최근 시작</label>
          <input class="form-control" type="date" id="recent_start" name="recent_start" value="{{ recent_start }}">
        </div>
        <div class="col-lg-2 col-md-6">
          <label class="form-label" for="recent_end">최근 종료</label>
          <input class="form-control" type="date" id="recent_end" name="recent_end" value="{{ recent_end }}">
        </div>
        <div class="col-lg-1 col-md-12 d-grid">
          <button type="submit" class="btn btn-primary" style="background: var(--accent); border-color: var(--accent);">보기</button>
        </div>
      </form>
      <div class="mt-3 text-secondary small">0.00 값은 결측치로 처리하며, 선택한 기간에 데이터가 없으면 안내 문구를 보여줍니다.</div>
    </section>

    {% if no_data_warning %}
    <div class="alert alert-warning border-0 shadow-sm">{{ no_data_warning }}</div>
    {% endif %}

    <section class="mb-3">
      <div class="row g-3">
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
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">차이 금액</div>
            <div class="kpi-value">{{ diff_display }}</div>
            <div class="kpi-sub">달러</div>
          </div>
        </div>
        <div class="col-md-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-label">증감률</div>
            <div class="kpi-value">{{ percent_display }}</div>
            <div class="kpi-sub">이전 기간 평균 기준</div>
          </div>
        </div>
      </div>
    </section>

    <section class="mb-4">
      <div class="fw-bold mb-2">핵심 문구</div>
      <div class="mb-2">{{ summary_line_1 }}</div>
      <div class="mb-2">{{ summary_line_2 }}</div>
      <div class="note-box">
        <div class="fw-bold mb-1">{{ recommendation_tag }}</div>
        <div>{{ recommendation_text }}</div>
      </div>
    </section>

    <section class="mb-4">
      <div class="fw-bold mb-2">시각화</div>
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
    default_recent_end = max_date
    default_recent_start = max(min_date, default_recent_end - timedelta(days=29))
    default_previous_end = max(min_date, default_recent_start - timedelta(days=1))
    default_previous_start = max(min_date, default_previous_end - timedelta(days=29))

    selected_oil = request.args.get("oil", "dubai")
    if selected_oil not in OIL_LABELS:
        selected_oil = "dubai"

    previous_start = safe_date(request.args.get("previous_start"), default_previous_start)
    previous_end = safe_date(request.args.get("previous_end"), default_previous_end)
    recent_start = safe_date(request.args.get("recent_start"), default_recent_start)
    recent_end = safe_date(request.args.get("recent_end"), default_recent_end)

    previous_avg, previous_period = summarize_period(df, selected_oil, previous_start, previous_end)
    recent_avg, recent_period = summarize_period(df, selected_oil, recent_start, recent_end)

    no_data_warning = None
    if previous_period.empty or recent_period.empty:
        no_data_warning = "해당 기간의 원유가격 데이터가 없습니다."

    diff = None
    percent_change = None
    if previous_avg is not None and recent_avg is not None:
        diff = recent_avg - previous_avg
        if previous_avg != 0:
            percent_change = diff / previous_avg * 100

    summary_line_1 = (
        f"이전 기간 평균 원유가격은 {format_price(previous_avg)}달러, "
        f"최근 기간 평균 원유가격은 {format_price(recent_avg)}달러입니다."
    )
    if percent_change is None:
        summary_line_2 = "이전 기간 평균이 0이거나 계산할 수 없어 증감률은 생략했습니다."
    elif percent_change > 0:
        summary_line_2 = f"최근 기간은 이전 기간 대비 {abs(percent_change):.2f}% 상승했습니다."
    elif percent_change < 0:
        summary_line_2 = f"최근 기간은 이전 기간 대비 {abs(percent_change):.2f}% 하락했습니다."
    else:
        summary_line_2 = "최근 기간은 이전 기간과 거의 변동이 없습니다."

    line_df = df.melt(id_vars="date", value_vars=OIL_COLUMNS, var_name="원유", value_name="가격")
    line_df["원유"] = line_df["원유"].map(lambda x: x.upper())
    line_df = line_df.dropna(subset=["가격"])

    line_fig = px.line(
        line_df,
        x="date",
        y="가격",
        color="원유",
        markers=True,
        title="날짜별 Dubai, Brent, WTI 원유가격 추이",
    )
    line_fig.update_layout(legend_title_text="", hovermode="x unified")
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
        title=f"{OIL_LABELS[selected_oil]} 평균 가격 비교",
    )
    bar_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    bar_fig.update_layout(showlegend=False, yaxis_title="달러")
    bar_chart = bar_fig.to_html(full_html=False, include_plotlyjs=False)

    return render_template_string(
        HTML_TEMPLATE,
        oil_options=OIL_LABELS,
        selected_oil=selected_oil,
        previous_start=previous_start.isoformat(),
        previous_end=previous_end.isoformat(),
        recent_start=recent_start.isoformat(),
        recent_end=recent_end.isoformat(),
        no_data_warning=no_data_warning,
        previous_avg_display=format_price(previous_avg),
        recent_avg_display=format_price(recent_avg),
        diff_display=format_price(diff),
        percent_display=format_change(percent_change),
        summary_line_1=summary_line_1,
        summary_line_2=summary_line_2,
        recommendation_tag=recommendation_tag(percent_change),
        recommendation_text=recommendation_text(percent_change),
        line_chart=line_chart,
        bar_chart=bar_chart,
    )


if __name__ == "__main__":
    app.run(debug=True)

