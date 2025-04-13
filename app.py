import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import tempfile
import os
import pandas as pd

def process_test_image(image_path, answer_key, passing_grade):
    """
    Process the test image and grade it with customizable answer key and passing grade
    """
    # Define the answer key from settings
    ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']

    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Find document contour
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    # If no document found
    if docCnt is None:
        return None, "Could not detect exam paper. Please ensure the image is clear."

    # Transform perspective
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # Thresholding
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find question contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Filter contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    # Sort contours
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    
    # Grade the exam
    correct = 0
    graded_paper = paper.copy()
    
    # To store detailed results
    detailed_results = []

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (0, 0, 255)  # Red
        k = answer_key[q]  # Use the custom answer key

        is_correct = k == bubbled[1]
        if is_correct:
            color = (0, 255, 0)  # Green
            correct += 1

        cv2.drawContours(graded_paper, [cnts[k]], -1, color, 3)
        cv2.putText(graded_paper, f"Q{q+1}", 
                    (cnts[0][0][0][0]-30, cnts[0][0][0][1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Store detailed results
        detailed_results.append({
            'question': q + 1,
            'correct_answer': ANSWER_LABELS[k],
            'student_answer': ANSWER_LABELS[bubbled[1]],
            'is_correct': is_correct
        })

    # Calculate score
    score = (correct / 5.0) * 100
    cv2.putText(graded_paper, f"Score: {score:.2f}%", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Determine pass/fail status
    pass_status = score >= passing_grade

    return graded_paper, f"{score:.2f}%", detailed_results, pass_status

def main():
    # Set page configuration
    st.set_page_config(
        page_title="🖋️ Test Grader App",
        page_icon="📝",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .title {
        text-align: center;
        color: #1E90FF;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title">🖋️ Multiple Choice Test Grader</div>', unsafe_allow_html=True)

    # Sidebar instructions
    st.sidebar.header("📝 Test Grading Instructions")
    st.sidebar.markdown("""
    1. Upload a clear image of the test
    2. Ensure the test is on a plain background
    3. Make sure bubbles are fully filled
    4. Image should be high-resolution
    """, unsafe_allow_html=True)

    # Initialize session state for tracking
    if 'graded' not in st.session_state:
        st.session_state.graded = False
    
    # Initialize default settings in session state
    if 'answer_key' not in st.session_state:
        st.session_state.answer_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
    if 'passing_grade' not in st.session_state:
        st.session_state.passing_grade = 60.0

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Test", "📊 Test Results", "⚙️ Settings"])

    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Your Test Image", 
            type=['png', 'jpg', 'jpeg'], 
            help="Upload a scanned multiple-choice test"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process the image
            try:
                with st.spinner('Grading the test...'):
                    graded_paper, score, detailed_results, pass_status = process_test_image(
                        tmp_path, 
                        st.session_state.answer_key, 
                        st.session_state.passing_grade
                    )

                # Store results in session state
                st.session_state.graded_paper = cv2.cvtColor(graded_paper, cv2.COLOR_BGR2RGB)
                st.session_state.original_image = uploaded_file
                st.session_state.score = score
                st.session_state.detailed_results = detailed_results
                st.session_state.pass_status = pass_status
                st.session_state.graded = True

                # Success message
                st.success("Test graded successfully!")

                # Automatically switch to results tab
                 st.query_params

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            # Clean up temporary file
            finally:
                os.unlink(tmp_path)

    with tab2:
        # Check if grading has been done
        if st.session_state.graded:
            # Columns for images
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Test")
                st.image(st.session_state.original_image, use_container_width=True)

            with col2:
                st.subheader("Graded Test")
                st.image(st.session_state.graded_paper, use_container_width=True)

            # Score display with pass/fail status
            pass_color = 'green' if st.session_state.pass_status else 'red'
            pass_text = 'PASS' if st.session_state.pass_status else 'FAIL'
            st.markdown(f'<div class="big-font">🏆 Test Score: {st.session_state.score} <span style="color:{pass_color}">{pass_text}</span></div>', unsafe_allow_html=True)

            # Detailed Breakdown Section
            st.subheader("📋 Detailed Question Breakdown")
            
            # Create a container for detailed breakdown
            breakdown_container = st.container()
            
            with breakdown_container:
                # Table for detailed results
                df_data = []
                for result in st.session_state.detailed_results:
                    df_data.append({
                        'Question': result['question'],
                        'Correct Answer': result['correct_answer'],
                        'Your Answer': result['student_answer'],
                        'Result': '✓ Correct' if result['is_correct'] else '✗ Incorrect'
                    })
                
                # Convert to DataFrame and style
                df = pd.DataFrame(df_data)
                
                # Color coding for results
                def color_result(val):
                    color = 'green' if '✓' in str(val) else 'red'
                    return f'color: {color}'
                
                styled_df = df.style.applymap(color_result)
                st.dataframe(styled_df, use_container_width=True)

                # Additional statistics
                total_questions = len(df)
                correct_questions = sum(1 for result in st.session_state.detailed_results if result['is_correct'])
                
                st.markdown(f"""
                ### 📊 Performance Insights
                - **Total Questions:** {total_questions}
                - **Correct Answers:** {correct_questions}
                - **Incorrect Answers:** {total_questions - correct_questions}
                """)
        else:
            st.info("Please upload a test image in the 'Upload Test' tab first.")

    with tab3:
        st.header("⚙️ Test Grading Settings")
        
        # Passing Grade Configuration
        st.subheader("🎯 Passing Grade")
        st.session_state.passing_grade = st.slider(
            "Set Passing Grade Percentage", 
            min_value=0.0, 
            max_value=100.0, 
            value=st.session_state.passing_grade, 
            step=0.5
        )
        st.info(f"Current Passing Grade: {st.session_state.passing_grade}%")

        # Answer Key Configuration
        st.subheader("✏️ Answer Key Configuration")
        
        # Create input fields for each question
        answer_labels = ['A', 'B', 'C', 'D', 'E']
        for q in range(5):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Question {q+1}")
            with col2:
                # Use radio button for selecting correct answer
                selected_answer = st.radio(
                    f"Select Correct Answer for Q{q+1}", 
                    answer_labels, 
                    index=st.session_state.answer_key[q],
                    key=f"answer_q{q}",
                    horizontal=True
                )
                
                # Update answer key in session state
                st.session_state.answer_key[q] = answer_labels.index(selected_answer)

        # Save Settings Button
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

    # About section
    st.markdown("---")
    st.markdown("### 💡 About This App")
    st.markdown("""
    - Automatically grades multiple-choice tests
    - Uses computer vision techniques
    - Supports standard 5-question multiple-choice tests
    - Works best with clear, high-contrast images
    - Customizable answer key and passing grade
    """)

# Run the app
if __name__ == "__main__":
    main()
