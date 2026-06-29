# app.py

import streamlit as st
import pandas as pd
from PIL import Image

# paste/import all your extraction helper functions here
# including: extract_table_from_image(pil_img)

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
