import re
import difflib
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image


# -----------------------------
# Streamlit compatibility helpers
# -----------------------------

def st_image_compat(img, **kwargs):
    """Support both newer and older Streamlit args for st.image."""
    try:
        return st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        return st.image(img, use_column_width=True, **kwargs)

def st_dataframe_compat(df, **kwargs):
    """Support both newer and older Streamlit args for st.dataframe."""
    try:
        return st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)


# -----------------------------
# OCR + Table extraction helpers (robust for 294 variants)
# -----------------------------

def _norm_token(s: str) -> str:
    """Normalize OCR token for matching."""
    s = str(s).lower()
    s = s.translate(str.maketrans({
        "|": "1",
        "i": "1",
        "l": "1",
        "o": "0",
    }))
    s = s.replace("1", "l").replace("0", "o")
    s = re.sub(r"[^a-z0-9%()]+", "", s)
    return s

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

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
    max_token_gap: int = 3,
) -> Optional[Dict[str, int]]:
    """
    Fuzzy-ish phrase matcher:
    - Searches optionally within top y_max (None = whole image)
    - Matches required_tokens in order on the same OCR line, allowing small gaps
    - Uses light OCR normalization
    """
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()

    d["t"] = d["text"].map(_norm_token)

    for _, grp in d.groupby(["block_num", "par_num", "line_num"]):
        words = grp.sort_values("left").reset_index(drop=True)
        toks = words["t"].tolist()

        idxs = []
        j = 0
        for req in required_tokens:
            found = False
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

def find_pair_box_similar(
    df: pd.DataFrame,
    w1: str,
    w2: str,
    y_max: Optional[int] = None,
    sim_thresh: float = 0.75,
    max_gap: int = 8,
) -> Optional[Dict[str, int]]:
    """Find two-word header like 'total students' even if OCR is slightly wrong."""
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
    sim_thresh: float = 0.72,
) -> Optional[Dict[str, int]]:
    """Find a single token header like 'nodata' even if OCR is slightly wrong."""
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()
    d["t"] = d["text"].map(_norm_token)

    best_row = None
    best_score = 0.0
    for _, r in d.iterrows():
        sc = _sim(r["t"], target)
        if sc > best_score:
            best_score = sc
            best_row = r

    if best_row is not None and best_score >= sim_thresh:
        left = int(best_row["left"])
        top = int(best_row["top"])
        right = int(best_row["left"] + best_row["width"])
        bottom = int(best_row["top"] + best_row["height"])
        return {"left": left, "top": top, "right": right, "bottom": bottom}

    return None

def find_pair_box_similar_line(
    df: pd.DataFrame,
    w1: str,
    w2: str,
    y_max: Optional[int] = None,
    sim_thresh: float = 0.70,
    max_gap: int = 10,
) -> Optional[Dict[str, int]]:
    """Similarity-based two-token header finder (same line)."""
    d = df.copy()
    if y_max is not None:
        d = d[d["top"] < y_max].copy()
    d["t"] = d["text"].map(_norm_token)

    for _, grp in d.groupby(["block_num", "par_num", "line_num"]):
        words = grp.sort_values("left").reset_index(drop=True)
        toks = words["t"].tolist()
        for i, t in enumerate(toks):
            if _sim(t, w1) >= sim_thresh:
                for j in range(i + 1, min(len(toks), i + 1 + max_gap)):
                    if _sim(toks[j], w2) >= sim_thresh:
                        sel = words.iloc[[i, j]]
                        return {
                            "left": int(sel["left"].min()),
                            "top": int(sel["top"].min()),
                            "right": int((sel["left"] + sel["width"]).max()),
                            "bottom": int((sel["top"] + sel["height"]).max()),
                        }
    return None


def infer_official_class_range_from_body(df_words: pd.DataFrame, pad: int = 120) -> Optional[Tuple[int, int]]:
    """
    If we can't find the 'Official class' header, infer the column range by finding
    3-digit class codes in the table body and taking the leftmost dense cluster.
    """
    d = df_words.copy()
    d["text"] = d["text"].astype(str)

    # Find likely class codes (3 digits, common OCR corrections happen later)
    code_mask = d["text"].str.contains(r"\b\d{3}\b", regex=True, na=False)
    codes = d[code_mask].copy()
    if codes.empty:
        return None

    # Use x-center to find the leftmost cluster of codes
    codes["xc"] = codes["left"] + codes["width"] / 2
    # take leftmost 30% of code positions as "class column candidates"
    cutoff = np.quantile(codes["xc"], 0.30)
    candidates = codes[codes["xc"] <= cutoff]
    if candidates.empty:
        candidates = codes.nsmallest(min(len(codes), 50), "xc")

    left = int(candidates["left"].min()) - pad
    right = int((candidates["left"] + candidates["width"]).max()) + pad
    return (left, right)

def get_column_ranges(df_words: pd.DataFrame) -> Tuple[Dict[str, Tuple[int, int]], int]:
    dfw = df_words.copy()
    dfw["text"] = dfw["text"].astype(str)
    dfw["t"] = dfw["text"].map(_norm_token)

    # Find the actual table header row by locating the line with Official class + Total students
    header_line = None

    for _, grp in dfw.groupby(["block_num", "par_num", "line_num"]):
        line = grp.sort_values("left").copy()
        toks = line["t"].tolist()
        joined = " ".join(toks)

        has_official = "official" in toks and "class" in toks
        has_total = "total" in toks and ("students" in toks or "student" in toks)

        if has_official and has_total:
            header_line = line
            break

    if header_line is None:
        raise ValueError("Could not find the table header row.")

    def box_for_tokens(tokens):
        toks = header_line["t"].tolist()

        for i in range(len(toks)):
            j = i
            matched = []

            for token in tokens:
                found = False
                while j < len(toks):
                    if _sim(toks[j], token) >= 0.70:
                        matched.append(j)
                        j += 1
                        found = True
                        break
                    j += 1

                if not found:
                    matched = []
                    break

            if matched:
                sel = header_line.iloc[matched]
                return {
                    "left": int(sel["left"].min()),
                    "top": int(sel["top"].min()),
                    "right": int((sel["left"] + sel["width"]).max()),
                    "bottom": int((sel["top"] + sel["height"]).max()),
                }

        return None

    official_box = box_for_tokens(["official", "class"])
    ongrade_box = box_for_tokens(["on", "grade", "level"])
    nodata_box = box_for_tokens(["nodata"]) or box_for_tokens(["no", "data"])
    total_box = box_for_tokens(["total", "students"]) or box_for_tokens(["total", "student"])

    missing = []
    if official_box is None:
        missing.append("official_class")
    if ongrade_box is None:
        missing.append("on_grade")
    if nodata_box is None:
        missing.append("no_data")
    if total_box is None:
        missing.append("total_students")

    if missing:
        raise ValueError(f"Could not find required header(s): {', '.join(missing)}")

    pad = 140

    ranges = {
        "official_class": (official_box["left"] - pad, official_box["right"] + pad),
        "on_grade": (ongrade_box["left"] - pad, ongrade_box["right"] + pad),
        "no_data": (nodata_box["left"] - pad, nodata_box["right"] + pad),
        "total_students": (total_box["left"] - pad, total_box["right"] + pad),
    }

    y_start = max(
        official_box["bottom"],
        ongrade_box["bottom"],
        nodata_box["bottom"],
        total_box["bottom"],
    ) + 20

    return ranges, y_start

def cluster_rows(df: pd.DataFrame, y_start: int, y_gap: int = 30) -> List[pd.DataFrame]:
    """Cluster word boxes into table rows using y-centers and a gap threshold."""
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
    s = str(text).strip()

    # OCR sometimes reads 0% as O% or Ox
    s = s.replace("O", "0").replace("o", "0").replace("x", "%")

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) / 100.0 if m else None

def normalize_class_code(text: str) -> Optional[str]:
    """Extract a 3-digit class code with light OCR error correction."""
    t = str(text).strip()
    if not t:
        return None

    first = t.split()[0]
    trans = str.maketrans({
        "$": "5",
        "S": "5",
        "s": "5",
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
    })
    first = first.translate(trans)

    m = re.search(r"(\d{3})", first)
    if m:
        return m.group(1)

    tt = t.translate(trans)
    m = re.search(r"(\d{3})", tt)
    return m.group(1) if m else None

def extract_table_from_image(pil_img: Image.Image) -> pd.DataFrame:
    thr = preprocess_for_ocr(pil_img, scale=3.0)
    words = ocr_to_dataframe(thr)

    if words.empty:
        return pd.DataFrame(columns=["class_code", "proficiency", "n_size"])

    words = words.copy()
    words["text"] = words["text"].astype(str)
    words["xc"] = words["left"] + words["width"] / 2
    words["yc"] = words["top"] + words["height"] / 2

    # Only table body, not chart
    table_words = words[words["top"] > words["top"].quantile(0.45)].copy()

    # Only class codes in the far-left table column
    anchors = table_words[
        table_words["text"].str.match(r"^\d{3}$", na=False)
    ].copy()

    if anchors.empty:
        return pd.DataFrame(columns=["class_code", "proficiency", "n_size"])

    # Keep only leftmost class-code anchors, ignoring chart axis labels
    left_cutoff = anchors["left"].quantile(0.25) + 100
    anchors = anchors[anchors["left"] <= left_cutoff].copy()

    # Column centers from your table layout, scaled OCR coordinates
    # These are: proficiency, mid, early, one below, two below, three+ below, no data, total students
    col_centers = {
        "proficiency": 2280,
        "mid": 2769,
        "early": 3261,
        "one_below": 3752,
        "two_below": 4245,
        "three_below": 4736,
        "no_data": 5228,
        "total_students": 5721,
    }

    tolerance = 110
    records = []

    for _, anchor in anchors.iterrows():
        class_code = normalize_class_code(anchor["text"])
        if not class_code:
            continue

        rowdf = table_words[
            (table_words["yc"] >= anchor["yc"] - 22) &
            (table_words["yc"] <= anchor["yc"] + 22)
        ].copy()

        row_values = {}

        for col, center in col_centers.items():
            cell = rowdf[
                (rowdf["xc"] >= center - tolerance) &
                (rowdf["xc"] <= center + tolerance)
            ].sort_values("left")

            cell_text = " ".join(cell["text"].astype(str).tolist()).strip()
            row_values[col] = cell_text

        proficiency = parse_percent(row_values["proficiency"])

        # Missing count cells are usually skipped zeroes, so treat missing as 0
        mid = parse_int(row_values["mid"]) or 0
        early = parse_int(row_values["early"]) or 0
        one_below = parse_int(row_values["one_below"]) or 0
        two_below = parse_int(row_values["two_below"]) or 0
        three_below = parse_int(row_values["three_below"]) or 0

        n_size = mid + early + one_below + two_below + three_below

        records.append({
            "class_code": str(class_code),
            "proficiency": proficiency,
            "n_size": int(n_size),
        })

    out = pd.DataFrame(records)

    if not out.empty:
        out = out.drop_duplicates(subset=["class_code"], keep="first").reset_index(drop=True)

    return out


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Image-Table Extractor", layout="wide")
st.title("PNG Table Extractor")

with st.sidebar:
    st.header("Dataset-level fields")
    district = st.text_input("District", value="")
    assessment = st.selectbox("Assessment", ("Acadience","iReady","MAP","PELI"))
    content = st.selectbox("Content area", ("ELA","Pre-K","Math"))
    period = st.selectbox("Time of Year",("Beginning","Mid","End"))

st.subheader("Upload Images")
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one image to begin.")
    st.stop()


st.subheader("File fields")

# Store per-file metadata in session_state so it persists between reruns
if "file_meta" not in st.session_state:
    st.session_state["file_meta"] = {}

# Build metadata inputs for each uploaded file
for idx, f in enumerate(uploaded_files, start=1):
    fname = f.name
    if fname not in st.session_state["file_meta"]:
        st.session_state["file_meta"][fname] = {"site": ""}

    with st.expander(f"{idx}. {fname}", expanded=(idx == 1)):
        st.session_state["file_meta"][fname]["site"] = st.text_input(
            "Site name",
            value=st.session_state["file_meta"][fname]["site"],
            key=f"site_{fname}",
        )
        


st.divider()
st.subheader("Run extraction")

extract_all = st.button("Extract all files", type="primary")

all_outputs = []

if extract_all:
    # Validate that all sites are filled
    missing_sites = [fname for fname, meta in st.session_state["file_meta"].items() if not meta.get("site", "").strip()]
    if missing_sites:
        st.error(
            "Please enter a Site name for every file before extracting.\n\nMissing for:\n- "
            + "\n- ".join(missing_sites)
        )
    else:
        progress = st.progress(0)
        status = st.empty()

        for i, f in enumerate(uploaded_files, start=1):
            fname = f.name
            meta = st.session_state["file_meta"][fname]
            site = meta["site"]

            status.write(f"Extracting {i}/{len(uploaded_files)}: **{fname}**")
            try:
                pil_img = Image.open(f)
                extracted = extract_table_from_image(pil_img)

                # Attach dataset-level + per-file metadata
                extracted.insert(0, "district", str(district))
                extracted.insert(1, "assessment", str(assessment))
                extracted.insert(2, "content", str(content))
                extracted.insert(3, "site", str(site))
                extracted.insert(4, "period", str(period))

                # Ensure requested types
                extracted["district"] = extracted["district"].astype(str)
                extracted["assessment"] = extracted["assessment"].astype(str)
                extracted["content"] = extracted["content"].astype(str)
                extracted["site"] = extracted["site"].astype(str)
                extracted["period"] = extracted["period"].astype(str)
                extracted["class_code"] = extracted["class_code"].astype(str)
                extracted["n_size"] = extracted["n_size"].astype("Int64")

                all_outputs.append(extracted)

            except Exception as e:
                st.error(f"Extraction failed for {fname}: {e}")

            progress.progress(i / len(uploaded_files))

        status.empty()

        if all_outputs:
            st.success(f"Finished extracting {len(all_outputs)} file(s).")


st.subheader("Combined output")

if all_outputs:
    combined = pd.concat(all_outputs, ignore_index=True)

    col_order = ["district", "assessment", "content", "site", "period", "class_code", "proficiency", "n_size"]
    combined = combined[[c for c in col_order if c in combined.columns]]

    view2 = combined.assign(
        proficiency_pct=combined["proficiency"].map(lambda x: f"{x:.0%}" if pd.notna(x) else None)
    )
    st_dataframe_compat(view2)

    csv_bytes = combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="extracted_table_data.csv",
        mime="text/csv",
    )
else:
    st.info("Click **Extract all files** to run extraction.")
