import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from datetime import timedelta


st.set_page_config(
    page_title="유타",
    page_icon="✈️",
    layout="wide",
)


DATA_PATH = "data/oil_prices.csv"
OIL_COLUMNS = ["dubai", "brent", "wti"]
OIL_LABELS = {
    "Dubai": "dubai",
    "Brent": "brent",
    "WTI": "wti",
    "전체 평균": "all",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in OIL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0.0, col] = np.nan

    df["all"] = df[OIL_COLUMNS].mean(axis=1, skipna=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def get_series(df: pd.DataFrame, label: str) -> pd.Series:
    if label == "all":
        return df["all"]
    return df[label]


def summarize_period(df: pd.DataFrame, label: str, start, end):
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    period = df.loc[mask, ["date", label]].dropna(subset=[label])
    if period.empty:
        return None, period
    return float(period[label].mean()), period


def format_price(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def format_change(value):
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def recommendation(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return "판단 보류"
    if percent_change > 3:
        return "원유가격 기준으로는 지금 발권을 검토하는 것이 유리해 보입니다."
    if percent_change < -3:
        return "원유가격 기준으로는 조금 기다려보는 것도 선택지가 될 수 있습니다."
    return "원유가격 기준으로는 발권 시점 차이가 크지 않아 보입니다."


def recommendation_tag(percent_change):
    if percent_change is None or pd.isna(percent_change):
        return "판단 보류"
    if percent_change > 3:
        return "지금 발권 검토"
    if percent_change < -3:
        return "조금 더 대기"
    return "차이 크지 않음"


st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, #102542 0%, #1e5f74 52%, #79a6bf 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        line-height: 1.5;
        opacity: 0.95;
    }
    .kpi {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: #f7f9fc;
        border: 1px solid #e5eaf2;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #516072;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #102542;
        line-height: 1;
    }
    .kpi-sub {
        margin-top: 0.35rem;
        color: #66758a;
        font-size: 0.9rem;
    }
    .note-box {
        padding: 0.9rem 1rem;
        border-radius: 0.9rem;
        background: #fff8e7;
        border: 1px solid #f1d99f;
        color: #6c5200;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>원유가격 기반 일본행 항공권 발권 타이밍 참고 서비스</h1>
        <p>이 서비스는 항공권 가격이나 실제 유류할증료를 예측하지 않습니다. 오피넷 국제 원유가격 추이를 바탕으로 발권 타이밍을 참고할 수 있게 돕는 초간단 MVP입니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("data/oil_prices.csv 파일을 찾을 수 없습니다.")
    st.stop()

if df.empty:
    st.error("원유가격 데이터가 비어 있습니다.")
    st.stop()

min_date = df["date"].dt.date.min()
max_date = df["date"].dt.date.max()

default_recent_end = max_date
default_recent_start = max(min_date, default_recent_end - timedelta(days=29))
default_previous_end = max(min_date, default_recent_start - timedelta(days=1))
default_previous_start = max(min_date, default_previous_end - timedelta(days=29))

col_a, col_b = st.columns(2)

with col_a:
    oil_name = st.selectbox("원유 종류", list(OIL_LABELS.keys()), index=0)

with col_b:
    st.caption("0.00 값은 결측치로 처리합니다.")

st.subheader("기간 선택")
period_col_1, period_col_2 = st.columns(2)

with period_col_1:
    previous_start, previous_end = st.date_input(
        "이전 기간",
        value=(default_previous_start, default_previous_end),
        min_value=min_date,
        max_value=max_date,
    )

with period_col_2:
    recent_start, recent_end = st.date_input(
        "최근 기간",
        value=(default_recent_start, default_recent_end),
        min_value=min_date,
        max_value=max_date,
    )

selected_label = OIL_LABELS[oil_name]

previous_avg, previous_period = summarize_period(df, selected_label, previous_start, previous_end)
recent_avg, recent_period = summarize_period(df, selected_label, recent_start, recent_end)

if previous_period.empty or recent_period.empty:
    st.warning("해당 기간의 원유가격 데이터가 없습니다.")

diff = None
percent_change = None
if previous_avg is not None and recent_avg is not None:
    diff = recent_avg - previous_avg
    if previous_avg != 0:
        percent_change = diff / previous_avg * 100

st.subheader("핵심 결과")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.markdown(
    f"""
    <div class="kpi">
        <div class="kpi-label">이전 기간 평균</div>
        <div class="kpi-value">{format_price(previous_avg)}</div>
        <div class="kpi-sub">달러</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col2.markdown(
    f"""
    <div class="kpi">
        <div class="kpi-label">최근 기간 평균</div>
        <div class="kpi-value">{format_price(recent_avg)}</div>
        <div class="kpi-sub">달러</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col3.markdown(
    f"""
    <div class="kpi">
        <div class="kpi-label">차이 금액</div>
        <div class="kpi-value">{format_price(diff)}</div>
        <div class="kpi-sub">달러</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col4.markdown(
    f"""
    <div class="kpi">
        <div class="kpi-label">증감률</div>
        <div class="kpi-value">{format_change(percent_change)}</div>
        <div class="kpi-sub">기준: 이전 기간 평균</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

summary_line_1 = f"이전 기간 평균 원유가격은 {format_price(previous_avg)}달러, 최근 기간 평균 원유가격은 {format_price(recent_avg)}달러입니다."
if percent_change is None:
    summary_line_2 = "이전 기간 평균이 0이거나 계산할 수 없어 증감률은 생략했습니다."
else:
    if percent_change > 0:
        summary_line_2 = f"최근 기간은 이전 기간 대비 {abs(percent_change):.2f}% 상승했습니다."
    elif percent_change < 0:
        summary_line_2 = f"최근 기간은 이전 기간 대비 {abs(percent_change):.2f}% 하락했습니다."
    else:
        summary_line_2 = "최근 기간은 이전 기간과 거의 변동이 없습니다."

st.markdown(f"**{summary_line_1}**")
st.markdown(f"**{summary_line_2}**")

st.markdown(
    f"""
    <div class="note-box">
        <strong>{recommendation_tag(percent_change)}</strong><br>
        {recommendation(percent_change)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("시각화")

line_df = df.melt(id_vars="date", value_vars=OIL_COLUMNS, var_name="원유", value_name="가격")
line_df["원유"] = line_df["원유"].str.upper()
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
st.plotly_chart(line_fig, use_container_width=True)

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
    title=f"{oil_name} 평균 가격 비교",
    color="기간",
)
bar_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
bar_fig.update_layout(showlegend=False, yaxis_title="달러")
st.plotly_chart(bar_fig, use_container_width=True)

st.subheader("참고 문구")
st.write(f"현재 선택한 원유 종류는 {oil_name}입니다. {recommendation(percent_change)}")
