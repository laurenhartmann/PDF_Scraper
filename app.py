import streamlit as st
import pandas as pd
import tabula
import tempfile
import re

st.title("Batch PDF Attendance Extractor")

uploaded_files = st.file_uploader(
    "Upload one or more attendance PDFs",
    type="pdf",
    accept_multiple_files=True
)

metadata = []

if uploaded_files:
    st.subheader("Enter metadata for each file:")

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Metadata for: {file.name}"):
            session_date = st.date_input(f"Session Date for {file.name}", key=f"date_{i}")
            grade_level = st.selectbox(
                f"Grade Level for {file.name}", list(range(0, 13)), key=f"grade_{i}"
            )
            group_number = st.selectbox(f"Group # for {file.name}", [1, 2], key=f"group_{i}")
            metadata.append({
                "file": file,
                "session_date": session_date,
                "grade_level": grade_level,
                "group_number": group_number
            })

    if st.button("Extract All Attendance Data"):
        all_data = []

        def title_case(name):
            return " ".join([n.capitalize() for n in name.split()])

        for entry in metadata:
            file = entry["file"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                pdf_path = tmp.name

            try:
                dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)

                for df in dfs:
                    for row in df.values:
                        text = " ".join(str(x) for x in row if pd.notna(x))

                        email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
                        name_parts = re.findall(r"\b[A-Z]{2,}(?:\s[A-Z]{2,})*\b", text)
                        first_name_match = re.findall(r"\b[A-Z][a-z]+\b", text)
                        sign_in_match = re.search(r"\d{1,2} \d{4}\s+\d{1,2}:\d{2}", text)

                        if email_match and name_parts and first_name_match:
                            email = email_match.group(0)
                            last_name = title_case(" ".join(name_parts))
                            first_name = title_case(first_name_match[0])
                            attendance = True if sign_in_match else False

                            all_data.append({
                                "Email": email,
                                "First Name": first_name,
                                "Last Name": last_name,
                                "Attended": attendance,
                                "Session Date": entry["session_date"],
                                "Grade Level": entry["grade_level"],
                                "Group #": entry["group_number"],
                                "File": file.name
                            })

            except Exception as e:
                st.warning(f"Could not process {file.name}: {e}")

        if all_data:
            final_df = pd.DataFrame(all_data)
            st.success(f"Extracted {len(final_df)} rows across {len(metadata)} files.")
            st.dataframe(final_df)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Combined CSV",
                data=csv,
                file_name="combined_attendance.csv",
                mime="text/csv"
            )
