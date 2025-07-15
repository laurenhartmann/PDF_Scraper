import streamlit as st
import pandas as pd
import pdfplumber
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
    f"Grade Level for {file.name}",
    ["K"] + [str(i) for i in range(1, 13)],
    key=f"grade_{i}"
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
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(
                        page.extract_text() for page in pdf.pages if page.extract_text()
                    )

                lines = text.splitlines()

                for line in lines:
                    email_match = re.search(r"[\w\.-]+@[\w\.-]+", line)
                    sign_in_match = re.search(r"\bJul\s+\d{1,2}\s+2025.*(?:AM|PM)", line)

                    last_names = re.findall(r"\b[A-Z]{2,}(?:\s[A-Z]{2,})*\b", line)
                    first_names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", line)

                    if email_match and last_names and first_names:
                        email = email_match.group(0)
                        last_name = title_case(" ".join(last_names))
                        first_name = title_case(first_names[0])
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
