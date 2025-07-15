import streamlit as st
import pandas as pd
import pdfplumber
import tempfile
import re

st.title("Batch PDF Attendance Extractor (No Java Required)")

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

                # Extract the workshop title
                title_match = re.search(r"Title:\s*(.+)", text)
                title = title_match.group(1).strip() if title_match else "Unknown Title"

                lines = text.splitlines()
                current_school = "Unknown School"

                for line in lines:
                    # Detect new school header line
                    school_match = re.match(r"La Joya ISD - (.+)", line)
                    if school_match:
                        current_school = school_match.group(1).strip()
                        continue

                    # Parse participant rows
                    email_match = re.search(r"[\w\.-]+@[\w\.-]+", line)
                    sign_in_match = re.search(r"\bJul\s+\d{1,2}\s+2025.*(?:AM|PM)", line)

                    if email_match:
                        email = email_match.group(0)
                        parts = line.split()

                        try:
                            email_index = parts.index(email)

                            # Assume structure: email | status | last | first
                            first_name = parts[email_index + 3]
                            last_name = parts[email_index + 2]

                            full_name = f"{first_name} {last_name}"
                            full_name = title_case(full_name)
                            first_name = full_name.split()[0]
                            last_name = " ".join(full_name.split()[1:])

                            attendance = True if sign_in_match else False

                            all_data.append({
                                "Email": email,
                                "First Name": first_name,
                                "Last Name": last_name,
                                "School": current_school,
                                "Attended": attendance,
                                "Session Date": entry["session_date"],
                                "Grade Level": entry["grade_level"],
                                "Group #": entry["group_number"],
                                "Workshop Title": title,
                                "File": file.name
                            })
                        except IndexError:
                            st.warning(f"Could not parse name data for line: {line}")

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
