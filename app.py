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
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

                lines = text.splitlines()

                for line in lines:
                    email_match = re.search(r"[\w\.-]+@[\w\.-]+", line)
                    sign_in_match = re.search(r"\bJul\s+\d{1,2}\s+2025.*AM|PM", line)

                    # Try to extract uppercase last name(s)
