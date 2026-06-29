st.subheader("File fields")

# Store per-file metadata in session_state so it persists between reruns
if "file_meta" not in st.session_state:
    st.session_state["file_meta"] = {}

period_options = ["Beginning", "Mid", "End"]

# Build metadata inputs for each uploaded file
for idx, f in enumerate(uploaded_files, start=1):
    key_base = f.name  # unique enough for your use case
    if key_base not in st.session_state["file_meta"]:
        st.session_state["file_meta"][key_base] = {"site": "", "period": period_options[0]}

    with st.expander(f"{idx}. {f.name}", expanded=(idx == 1)):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["file_meta"][key_base]["site"] = st.text_input(
                "Site name",
                value=st.session_state["file_meta"][key_base]["site"],
                key=f"site_{key_base}",
            )
        with col2:
            st.session_state["file_meta"][key_base]["period"] = st.selectbox(
                "Period",
                period_options,
                index=period_options.index(st.session_state["file_meta"][key_base]["period"]),
                key=f"period_{key_base}",
            )

        # Optional preview
        pil_img_preview = Image.open(f)
        st_image_compat(pil_img_preview, caption=f.name)


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
            period = meta["period"]

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
