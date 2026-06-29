# app.py

import streamlit as st
import pandas as pd
from PIL import Image
import re
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import difflib

def _norm_token(s: str) -> str:
    """
    Normalize OCR token for matching:
    - lowercase
    - fix common OCR confusions (I/l/1, O/0)
    - strip non-alphanumerics except %()
    """
    s = str(s).lower()
    s = s.translate(str.maketrans({
        "|": "1",  # unify to digits then fix
        "i": "1",  # treat i as 1 (later)
        "l": "1",  # treat l as 1 (later)
        "o": "0",  # unify o/0
    }))
    # now map digits back to letters for matching words
    s = s.replace("1", "l").replace("0", "o")
    s = re.sub(r"[^a-z0-9%()]+", "", s)
    return s

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def find_pair_box_similar(
    df: pd.DataFrame,
    w1: str,
    w2: str,
    y_max: Optional[int] = None,
    sim_thresh: float = 0.78,
    max_gap: int = 4,
) -> Optional[Dict[str, int]]:
    """
    Find two-word header like 'total students' even if OCR is slightly wrong.
    Matches within same OCR line and within max_gap token distance.
    """
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()
    d["t"] = d["text"].map(_norm_token)

    for _, grp in d.groupby(["block_num", "par_num", "line_num"]):
        words = grp.sort_values("left").reset_index(drop=True)
        toks = words["t"].tolist()

        idx1s = [i for i, t in enumerate(toks) if _sim(t, w1) >= sim_thresh]
        if not idx1s:
            continue

        for i in idx1s:
            for j in range(i + 1, min(len(toks), i + 1 + max_gap)):
                if _sim(toks[j], w2) >= sim_thresh:
                    sel = words.iloc[[i, j]]
                    left = int(sel["left"].min())
                    top = int(sel["top"].min())
                    right = int((sel["left"] + sel["width"]).max())
                    bottom = int((sel["top"] + sel["height"]).max())
                    return {"left": left, "top": top, "right": right, "bottom": bottom}

    return None

def find_single_box_similar(
    df: pd.DataFrame,
    target: str,
    y_max: Optional[int] = None,
    sim_thresh: float = 0.78,
) -> Optional[Dict[str, int]]:
    """
    Find a single token header like 'nodata' even if OCR is slightly wrong.
    """
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()
    d["t"] = d["text"].map(_norm_token)

    # pick best token in header-ish region
    best = None
    best_score = 0.0
    for _, r in d.iterrows():
        sc = _sim(r["t"], target)
        if sc > best_score:
            best_score = sc
            best = r

    if best is not None and best_score >= sim_thresh:
        left = int(best["left"])
        top = int(best["top"])
        right = int(best["left"] + best["width"])
        bottom = int(best["top"] + best["height"])
        return {"left": left, "top": top, "right": right, "bottom": bottom}

    return None

def preprocess_for_ocr(pil_img: Image.Image, scale: float = 3.0) -> np.ndarray:
    """
    More robust preprocessing for screenshots where text is small or in shaded cells.
    - Upscale
    - Grayscale
    - Adaptive threshold
    - Auto-invert if needed
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Light denoise to stabilize thresholding
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive threshold tends to beat Otsu on UI screenshots / shaded header cells
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,  # blockSize
        11   # C
    )

    # If the result is mostly white, invert (sometimes OCR prefers dark text on white)
    white_ratio = (thr > 0).mean()
    if white_ratio > 0.92:
        thr = 255 - thr

    return thr

def ocr_to_dataframe(thr_img: np.ndarray) -> pd.DataFrame:
    """
    Try multiple PSM modes and pick the result with the most recognized tokens.
    This helps when tables are not detected under one segmentation mode.
    """
    best_df = None
    best_n = -1

    # 6 = block of text, 4 = single column, 11 = sparse text
    for psm in [6, 4, 11, 12]:
        df = pytesseract.image_to_data(
            thr_img,
            output_type=pytesseract.Output.DATAFRAME,
            config=f"--oem 1 --psm {psm}",
        )
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip() != ""].copy()

        if len(df) > best_n:
            best_df = df
            best_n = len(df)

    return best_df if best_df is not None else pd.DataFrame(columns=["text"])
def find_phrase_box_fuzzy(
    df: pd.DataFrame,
    required_tokens: List[str],
    y_max: Optional[int] = None,
    max_token_gap: int = 2,
) -> Optional[Dict[str, int]]:
    """
    Fuzzy-ish phrase matcher:
    - Searches optionally within top y_max (set None to search entire image)
    - Matches required_tokens in order on the same line, allowing small gaps
    - Uses light OCR normalization (lowercase + strip punctuation)
    """
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()

    def norm(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r"[^a-z0-9%()]+", "", s)  # keep %, (), alnum
        return s

    d["t"] = d["text"].map(norm)

    # group by OCR line
    for _, grp in d.groupby(["block_num", "par_num", "line_num"]):
        words = grp.sort_values("left")
        toks = words["t"].tolist()

        # try to match required tokens in-order, allowing max_token_gap skips
        idxs = []
        j = 0
        for req in required_tokens:
            found = False
            # search forward with limited gap
            for k in range(j, min(len(toks), j + max_token_gap + 1)):
                if toks[k] == req:
                    idxs.append(k)
                    j = k + 1
                    found = True
                    break
            if not found:
                idxs = []
                break

        if idxs:
            sel = words.iloc[idxs]
            left = int(sel["left"].min())
            top = int(sel["top"].min())
            right = int((sel["left"] + sel["width"]).max())
            bottom = int((sel["top"] + sel["height"]).max())
            return {"left": left, "top": top, "right": right, "bottom": bottom}

    return None

def cluster_rows(df: pd.DataFrame, y_start: int, y_gap: int = 30) -> List[pd.DataFrame]:
    d = df[df["top"] > y_start].copy()
    d["yc"] = d["top"] + d["height"] / 2
    d = d.sort_values("yc")

    rows: List[pd.DataFrame] = []
    current = []
    last_y = None
    for _, r in d.iterrows():
        if last_y is None or (r["yc"] - last_y) <= y_gap:
            current.append(r)
        else:
            rows.append(pd.DataFrame(current))
            current = [r]
        last_y = r["yc"]
    if current:
        rows.append(pd.DataFrame(current))
    return rows

def parse_int(text: str) -> Optional[int]:
    m = re.search(r"\d+", str(text))
    return int(m.group()) if m else None

def parse_percent(text: str) -> Optional[float]:
    s = str(text)
    if "%" not in s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) / 100.0 if m else None

def normalize_class_code(text: str) -> Optional[str]:
    t = str(text).strip()
    if not t:
        return None
    first = t.split()[0]
    trans = str.maketrans({"$": "5","S": "5","s": "5","O": "0","o": "0","I": "1","l": "1","|": "1"})
    first = first.translate(trans)
    m = re.search(r"(\d{3})", first)
    if m:
        return m.group(1)
    tt = t.translate(trans)
    m = re.search(r"(\d{3})", tt)
    return m.group(1) if m else None

def get_column_ranges(df_words: pd.DataFrame) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """
    Robust header detection:
    - Official class: exact/fuzzy phrase
    - On grade level (%): exact/fuzzy phrase
    - No data: similarity match (no-data / nodata / etc.)
    - Total students: similarity match (total student(s) / totaI / etc.)
    """
    dfw = df_words.copy()
    dfw["text"] = dfw["text"].astype(str)

    # Reuse your earlier fuzzy phrase finder for these (usually stable)
    official_box = find_phrase_box_fuzzy(dfw, ["official", "class"], y_max=520, max_token_gap=4) \
                   or find_phrase_box_fuzzy(dfw, ["officialclass"], y_max=None, max_token_gap=6)

    ongrade_box = (
        find_phrase_box_fuzzy(dfw, ["on", "grade", "level", "(%)"], y_max=520, max_token_gap=4)
        or find_phrase_box_fuzzy(dfw, ["on", "grade", "level"], y_max=None, max_token_gap=6)
        or find_phrase_box_fuzzy(dfw, ["ongradelevel"], y_max=None, max_token_gap=6)
    )

    # These are the problematic ones for your 294 image, so use similarity
    # Try in header-ish region first, then whole image
    no_data_box = (
        find_pair_box_similar(dfw, "no", "data", y_max=520, sim_thresh=0.75, max_gap=5)
        or find_single_box_similar(dfw, "nodata", y_max=520, sim_thresh=0.72)
        or find_pair_box_similar(dfw, "no", "data", y_max=None, sim_thresh=0.75, max_gap=8)
        or find_single_box_similar(dfw, "nodata", y_max=None, sim_thresh=0.72)
    )

    total_students_box = (
        find_pair_box_similar(dfw, "total", "students", y_max=520, sim_thresh=0.75, max_gap=6)
        or find_pair_box_similar(dfw, "total", "student", y_max=520, sim_thresh=0.75, max_gap=6)
        or find_pair_box_similar(dfw, "total", "students", y_max=None, sim_thresh=0.75, max_gap=10)
        or find_pair_box_similar(dfw, "total", "student", y_max=None, sim_thresh=0.75, max_gap=10)
    )

    boxes = {
        "official_class": official_box,
        "on_grade": ongrade_box,
        "no_data": no_data_box,
        "total_students": total_students_box,
    }

    missing = [k for k, v in boxes.items() if v is None]
    if missing:
        raise ValueError(f"Could not find required header(s): {', '.join(missing)}")

    pad = 120  # widen slightly for header variability
    ranges = {k: (boxes[k]["left"] - pad, boxes[k]["right"] + pad) for k in boxes.keys()}
    y_start = max(v["bottom"] for v in boxes.values()) + 20
    return ranges, y_start

def crop_to_text_region(pil_img: Image.Image, scale: float = 3.0, pad: int = 30) -> Image.Image:
    """
    Uses a quick OCR pass to find where text is, then crops to that bounding region.
    Great for removing charts / whitespace around the table.
    """
    thr = preprocess_for_ocr(pil_img, scale=scale)
    words = ocr_to_dataframe(thr)

    # If OCR found very little, don't crop
    if words is None or words.empty or len(words) < 30:
        return pil_img

    # Bounding box around all detected text
    left = int(words["left"].min())
    top = int(words["top"].min())
    right = int((words["left"] + words["width"]).max())
    bottom = int((words["top"] + words["height"]).max())

    # Map back to original image coords (because we scaled before OCR)
    s = scale
    left = max(int(left / s) - pad, 0)
    top = max(int(top / s) - pad, 0)
    right = min(int(right / s) + pad, pil_img.size[0])
    bottom = min(int(bottom / s) + pad, pil_img.size[1])

    # Avoid tiny crops
    if (right - left) < 300 or (bottom - top) < 300:
        return pil_img

    return pil_img.crop((left, top, right, bottom))

def extract_table_from_image(pil_img: Image.Image) -> pd.DataFrame:
    pil_img = crop_to_text_region(pil_img, scale=3.0, pad=30)
    thr = preprocess_for_ocr(pil_img, scale=3.0)
    words = ocr_to_dataframe(thr)
    ranges, y_start = get_column_ranges(words)
    rows = cluster_rows(words, y_start=y_start, y_gap=30)

    records = []
    for rowdf in rows:
        rowdf = rowdf.copy()
        rowdf["xc"] = rowdf["left"] + rowdf["width"] / 2

        def text_in(col: str) -> str:
            l, r = ranges[col]
            sel = rowdf[(rowdf["xc"] >= l) & (rowdf["xc"] <= r)].sort_values("left")
            return " ".join(sel["text"].astype(str).tolist()).strip()

        class_code = normalize_class_code(text_in("official_class"))
        if not class_code:
            continue

        proficiency = parse_percent(text_in("on_grade"))
        total_students = parse_int(text_in("total_students"))
        no_data = parse_int(text_in("no_data")) or 0
        if total_students is None:
            continue

        records.append({
            "class_code": str(class_code),
            "proficiency": proficiency,          # proportion (e.g., 0.37)
            "n_size": int(total_students - no_data),
        })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.drop_duplicates(subset=["class_code"], keep="first").reset_index(drop=True)
    return out

st.set_page_config(page_title="PNG Table Extractor", layout="wide")

st.title("PNG Table Extractor")
st.write("Upload one or more PNG screenshots. The app will extract table rows and concatenate them into one dataset.")

uploaded_files = st.file_uploader(
    "Upload PNG files",
    type=["png"],
    accept_multiple_files=True
)

if uploaded_files:
    all_results = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert("RGB")

            df = extract_table_from_image(img)

            if df.empty:
                errors.append({
                    "file_name": uploaded_file.name,
                    "error": "No rows extracted"
                })
                continue

            df.insert(0, "source_file", uploaded_file.name)
            all_results.append(df)

        except Exception as e:
            errors.append({
                "file_name": uploaded_file.name,
                "error": str(e)
            })

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        st.subheader("Concatenated extracted dataset")
        st.dataframe(final_df, use_container_width=True)

        csv = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download concatenated CSV",
            data=csv,
            file_name="extracted_png_tables.csv",
            mime="text/csv"
        )

    if errors:
        st.subheader("Files with extraction issues")
        st.dataframe(pd.DataFrame(errors), use_container_width=True)

else:
    st.info("Upload PNG files to begin.")
