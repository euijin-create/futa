from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


DEFAULT_INPUT = Path("data/opinet_raw.csv")
DEFAULT_OUTPUT = Path("data/opinet_full.csv")
DATE_COLUMN_CANDIDATES = ("date", "기간", "날짜")
VALUE_COLUMNS = ["dubai", "brent", "wti"]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"CSV를 읽을 수 없습니다: {path} ({last_error})")


def parse_date(value) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    text = (
        text.replace("년", "-")
        .replace("월", "-")
        .replace("일", "")
        .replace(".", "-")
        .replace("/", "-")
        .replace(" ", "")
    )
    text = text.rstrip("-")

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if re.fullmatch(r"\d{2}-\d{1,2}-\d{1,2}", text):
        return pd.to_datetime(text, format="%y-%m-%d", errors="coerce")
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if re.fullmatch(r"\d{6}", text):
        return pd.to_datetime(text, format="%y%m%d", errors="coerce")

    return pd.to_datetime(text, errors="coerce")


def normalize_opinet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    date_col = next(
        (
            col
            for col in df.columns
            if col.lower() == "date" or any(token in col for token in DATE_COLUMN_CANDIDATES[1:])
        ),
        df.columns[0],
    )

    rename_map = {date_col: "date"}
    for col in df.columns:
        lower = str(col).strip().lower()
        if lower in VALUE_COLUMNS:
            rename_map[col] = lower

    df = df.rename(columns=rename_map)

    missing = [col for col in VALUE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"누락된 원유 가격 컬럼이 있습니다: {missing}")

    df["date"] = df["date"].map(parse_date)
    for col in VALUE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0.0, col] = pd.NA

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    ordered_columns = ["date"] + VALUE_COLUMNS
    return df[ordered_columns].reset_index(drop=True)


def print_validation(df: pd.DataFrame) -> None:
    dates = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = dates.dropna()
    broken_dates = int(dates.isna().sum())
    duplicate_dates = int(valid_dates.duplicated().sum())

    meta = {
        "firstDate": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        "lastDate": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
    }

    print(f"meta.firstDate: {meta['firstDate']}")
    print(f"meta.lastDate: {meta['lastDate']}")
    print(f"전체 행 수: {len(df)}")
    print(f"유효한 날짜 개수: {len(valid_dates)}")
    print(f"누락/깨진 날짜 개수: {broken_dates}")
    print(f"중복 날짜 개수: {duplicate_dates}")

    if meta["firstDate"] is None or meta["lastDate"] is None:
        raise ValueError("날짜 파싱 결과가 비어 있습니다.")
    if not meta["firstDate"].startswith("2008-"):
        raise ValueError(f"첫 날짜가 2008년대가 아닙니다: {meta['firstDate']}")
    if not meta["lastDate"].startswith("2026-"):
        raise ValueError(f"마지막 날짜가 2026년대가 아닙니다: {meta['lastDate']}")
    if broken_dates != 0:
        raise ValueError(f"깨진 날짜가 남아 있습니다: {broken_dates}")
    if duplicate_dates != 0:
        raise ValueError(f"중복 날짜가 남아 있습니다: {duplicate_dates}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize OPINET oil CSV into UTF-8 YYYY-MM-DD format.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="원본 CSV 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="정규화 CSV 경로")
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {args.input}")

    raw_df = read_csv_with_fallback(args.input)
    normalized_df = normalize_opinet(raw_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"saved: {args.output}")
    print_validation(normalized_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
