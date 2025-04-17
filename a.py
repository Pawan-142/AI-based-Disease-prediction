import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import plotly.graph_objects as go

# 🔥 MUST be the first Streamlit command
st.set_page_config(
    page_title="Health Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
/* Main styling */
.main-header {
    color: #2c3e50;
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #f1f1f1;
    font-family: 'Arial', sans-serif;
}
.sub-header {
    color: #2c3e50;
    font-size: 2rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    font-family: 'Arial', sans-serif;
}
.tab-subheader {
    color: #3498db;
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.positive-result {
    background-color: #ffe8e8;
    border: 2px solid #ff9999;
}
.negative-result {
    background-color: #e8ffe8;
    border: 2px solid #99ff99;
}
.info-box {
    background-color: #f0f7ff;
    border: 1px solid #bbd6ff;
    border-radius: 5px;
    padding: 15px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.treatment-box, .prevention-box {
    background-color: #f9f9f9;
    border-radius: 5px;
    padding: 15px;
    height: 100%;
    border-left: 4px solid #3498db;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.prevention-box {
    border-left-color: #2ecc71;
}
.cure-box {
    background-color: #f9f9f9;
    border-radius: 5px;
    padding: 15px;
    height: 100%;
    border-left: 4px solid #9b59b6;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
div.stButton > button {
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #f1f1f1;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #3498db;
    color: white;
}
table {
    font-size: 0.9rem;
}
thead tr th {
    background-color: #f1f1f1;
    font-weight: 600;
}
.footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #f1f1f1;
    font-size: 0.8rem;
    color: #7f8c8d;
}
.disease-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    height: 100%;
    transition: all 0.3s ease;
    border-left: 5px solid #3498db;
}
.disease-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.disease-card-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 10px;
}
.disease-card-desc {
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-bottom: 20px;
}
.disease-card-stat {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e74c3c;
}
.stNumberInput input {
    border-radius: 5px;
    border: 1px solid #e0e0e0;
}
.stNumberInput input:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}
.nav-card {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.nav-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}
.nav-card-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
}
.nav-card-icon {
    font-size: 1.5rem;
    margin-right: 10px;
    color: #3498db;
}
.stNumberInput > div > div > input {
    height: 40px;
}
.profile-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    border-top: 4px solid #3498db;
}
.stCheckbox label span {
    font-size: 1rem;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# --- Function to load trained models ---
@st.cache_resource
def load_models():
    """Load trained machine learning models for each disease."""
    models = {}
    
    # Check if models exist, and load them
    model_paths = {
        'diabetes': 'models/diabetes_model.pkl',
        'heart': 'models/heart_disease_model.pkl',
        'liver': 'models/liver_disease_model.pkl',
        'kidney': 'models/kidney_disease_model.pkl',
        'parkinsons': 'models/parkinsons_model.pkl',
    }
    
    for disease, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                models[disease] = pickle.load(f)
            st.sidebar.success(f"✅ {disease.capitalize()} model loaded successfully")
        except FileNotFoundError:
            st.sidebar.error(f"❌ {disease.capitalize()} model not found at {path}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {disease} model: {str(e)}")
    
    return models

# --- Disease information ---
disease_info = {
    'diabetes': {
        'name': 'Diabetes',
        'description': 'Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, or when the body cannot make good use of the insulin it produces.',
        'symptoms': ['Frequent urination', 'Increased thirst', 'Unexplained weight loss', 'Extreme hunger', 'Blurry vision', 'Fatigue', 'Slow-healing sores'],
        'risk_factors': ['Overweight', 'Family history', 'Physical inactivity', 'Age', 'High blood pressure', 'Abnormal cholesterol levels', 'History of gestational diabetes'],
        'treatments': ['Insulin therapy', 'Diet management', 'Regular exercise', 'Blood sugar monitoring', 'Oral medications', 'Regular foot care', 'Eye examinations', 'Kidney function tests'],
        'prevention': ['Maintain a healthy weight', 'Regular physical activity (150 minutes per week)', 'Healthy diet with plenty of fiber and whole grains', 'Limit refined carbohydrates and sugary foods', 'Stay hydrated with water instead of sugary beverages', 'Avoid smoking', 'Limit alcohol consumption', 'Regular check-ups', 'Stress management', 'Adequate sleep'],
        'cure': 'There is currently no cure for diabetes, but it can be managed effectively through lifestyle changes, medication, and regular monitoring. Type 2 diabetes can sometimes go into remission with significant weight loss and lifestyle changes, but ongoing management is still necessary. Research into artificial pancreas technology and islet cell transplantation shows promise for future treatment options.'
    },
    'heart': {
        'name': 'Heart Disease',
        'description': 'Heart disease describes a range of conditions that affect your heart, including coronary artery disease, heart rhythm problems (arrhythmias) and heart defects.',
        'symptoms': ['Chest pain or discomfort', 'Shortness of breath', 'Pain in the neck, jaw, throat, upper abdomen or back', 'Numbness in arms or legs', 'Fatigue', 'Irregular heartbeat', 'Dizziness or lightheadedness', 'Swelling in legs, ankles, or feet'],
        'risk_factors': ['Age', 'Sex', 'Family history', 'Smoking', 'High blood pressure', 'High cholesterol', 'Diabetes', 'Obesity', 'Physical inactivity', 'Stress', 'Poor diet', 'Excessive alcohol use'],
        'treatments': ['Medications (statins, beta-blockers, ACE inhibitors)', 'Surgery or medical procedures (angioplasty, stent placement, bypass surgery)', 'Lifestyle changes', 'Cardiac rehabilitation', 'Implantable devices (pacemakers, ICDs)', 'Heart valve repair or replacement', 'Heart transplant (in severe cases)'],
        'prevention': ['Regular exercise (at least 150 minutes of moderate activity weekly)', 'Heart-healthy diet rich in fruits, vegetables, and whole grains', 'Maintain healthy weight', 'Quit smoking and avoid secondhand smoke', 'Limit alcohol to 1-2 drinks per day', 'Manage stress through meditation, yoga, or other relaxation techniques', 'Regular check-ups and blood pressure monitoring', 'Manage existing health conditions like diabetes or high blood pressure', 'Get adequate sleep (7-8 hours)', 'Know your numbers (cholesterol, blood pressure, blood sugar)'],
        'cure': 'Heart disease generally cannot be cured completely, but it can be effectively managed and its progression can be slowed or halted with proper treatment. Some forms of heart disease may be improved through interventions like valve replacements or repair of congenital defects. The focus of treatment is typically on controlling symptoms, reducing risk factors, and preventing complications. Emerging research in stem cell therapy and genetic treatments offers hope for more effective treatments in the future.'
    },
    'liver': {
        'name': 'Liver Disease',
        'description': 'Liver disease is any disturbance of liver function that causes illness. It can be inherited or caused by various factors such as viruses and alcohol use.',
        'symptoms': ['Yellowish skin and eyes (jaundice)', 'Abdominal pain and swelling', 'Swelling in the legs and ankles', 'Itchy skin', 'Dark urine color', 'Pale stool color', 'Chronic fatigue', 'Nausea or vomiting', 'Loss of appetite', 'Tendency to bruise easily'],
        'risk_factors': ['Heavy alcohol use', 'Obesity', 'Type 2 diabetes', 'Tattoos or body piercings', 'Injecting drugs using shared needles', 'Blood transfusion before 1992', 'Exposure to certain chemicals or toxins', 'Family history of liver disease', 'Unprotected sex', 'High levels of triglycerides in blood'],
        'treatments': ['Lifestyle changes (reduced alcohol, healthy diet)', 'Medications to treat the underlying cause', 'Medications for symptom management', 'Liver transplant for advanced liver failure', 'Treatment for complications (varices, ascites)', 'Antiviral medications for viral hepatitis', 'Immunosuppressants for autoimmune hepatitis'],
        'prevention': ['Limit alcohol consumption (no more than 1-2 drinks per day)', 'Maintain a healthy weight through diet and exercise', 'Get vaccinated against hepatitis A and B', 'Use medications wisely and follow recommended dosages', 'Avoid contact with other peoples blood and body fluids', 'Practice safe sex', 'Dont share needles', 'Use protective gear when handling toxins', 'Regular screening if at high risk', 'Avoid mixing medications and alcohol'],
        'cure': 'The liver has remarkable regenerative capabilities, allowing it to heal from minor damage. Some liver conditions, like hepatitis A, can resolve completely with proper treatment and rest. For chronic conditions like hepatitis B, C, or cirrhosis, there are effective treatments but often not complete cures. Advanced liver disease may require a liver transplant. Early detection and intervention provide the best outcomes for liver disease management.'
    },
    'kidney': {
        'name': 'Kidney Disease',
        'description': 'Kidney disease is a condition in which the kidneys are damaged and cannot filter blood as well as they should. This can lead to waste buildup in the body.',
        'symptoms': ['Nausea', 'Vomiting', 'Loss of appetite', 'Fatigue and weakness', 'Sleep problems', 'Changes in urine output', 'Decreased mental sharpness', 'Muscle twitches and cramps', 'Swelling of feet and ankles', 'Persistent itching', 'Chest pain', 'Shortness of breath'],
        'risk_factors': ['Diabetes', 'High blood pressure', 'Heart disease', 'Family history of kidney disease', 'Age', 'Smoking', 'Obesity', 'Ethnicity (African Americans, Native Americans, and Asian Americans are at higher risk)', 'Chronic infections', 'Kidney stones'],
        'treatments': ['Medications to control blood pressure', 'Medications to lower cholesterol', 'Medications for anemia', 'Dialysis (hemodialysis or peritoneal dialysis)', 'Kidney transplant', 'Dietary changes (low-sodium, low-protein diets)', 'Erythropoietin supplements for anemia', 'Phosphate binders'],
        'prevention': ['Manage diabetes and high blood pressure through medication and lifestyle changes', 'Maintain a healthy weight through regular exercise and a balanced diet', 'Dont smoke or use tobacco products', 'Limit alcohol consumption', 'Stay hydrated but avoid excessive fluid intake', 'Follow a kidney-friendly diet low in sodium and protein if at risk', 'Monitor kidney function with regular tests if you have risk factors', 'Use caution with over-the-counter pain medications', 'Control blood cholesterol levels', 'Regular check-ups if you have kidney disease risk factors'],
        'cure': 'Chronic kidney disease typically has no cure, but treatments can help control symptoms and slow progression. Acute kidney injuries may heal completely with proper treatment. For end-stage kidney disease, dialysis or kidney transplantation are the main treatment options. Kidney transplantation offers the closest thing to a cure, though anti-rejection medications will be needed for life. Ongoing research in regenerative medicine and artificial kidneys offers hope for future treatment breakthroughs.'
    },

'parkinsons': {
    'name': "Parkinson's Disease",
    'description': "Parkinson's disease is a progressive nervous system disorder that affects movement.",
    'symptoms': ['Tremors', 'Slow movement', 'Stiffness', 'Balance problems', 'Speech changes'],
    'risk_factors': ['Age', 'Genetics', 'Environmental toxins', 'Gender (men are more likely)', 'Head trauma'],
    'treatments': ['Levodopa and carbidopa', 'Dopamine agonists', 'Physical therapy', 'Speech therapy', 'Surgical therapy like Deep Brain Stimulation'],
    'prevention': ['Regular exercise', 'Healthy diet', 'Avoiding exposure to pesticides or herbicides', 'Wearing helmets during activities', 'Monitoring for early symptoms'],
    'cure': 'There is currently no cure, but symptoms can be managed with medications, lifestyle changes, and therapy. Research in gene therapy and stem cells shows promise.'
},

}

# --- Main app functions ---
def predict_parkinsons_disease(fo, fhi, flo, jitter_percent, shimmer, rap, ppq, ddp, shimmer_db,
                                apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe, model):
    """Predict Parkinson's disease using the loaded model."""
    input_data = np.array([[
        fo, fhi, flo, jitter_percent, shimmer, rap, ppq, ddp, shimmer_db,
        apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
    ]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return prediction[0], probability

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, model):
    """Predict diabetes using the loaded model."""
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return prediction[0], probability

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, model):
    """Predict heart disease using the loaded model."""
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return prediction[0], probability

def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphatase, 
                         sgpt, sgot, total_proteins, albumin, ag_ratio, model):
    """Predict liver disease using the loaded model."""
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphatase, 
                         sgpt, sgot, total_proteins, albumin, ag_ratio]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return prediction[0], probability

def predict_kidney_disease(age, bp, sg, albumin, sugar, rbc, pc, pcc, bu, sc, sod, model):
    """Predict kidney disease using the loaded model."""
    input_data = np.array([[age, bp, sg, albumin, sugar, rbc, pc, pcc, bu, sc, sod]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return prediction[0], probability
def display_results(prediction, probability, disease_type):
    """Display prediction results with appropriate styling."""
    if prediction == 1:
        st.markdown(f'<div class="result-box positive-result">'
                    f'<h3>⚠️ High Risk of {disease_info[disease_type]["name"]} Detected</h3>'
                    f'<p>The model predicts a <strong>{probability:.1%}</strong> probability of {disease_info[disease_type]["name"]}.</p>'
                    f'<p>Please consult with a healthcare professional for a proper diagnosis.</p>'
                    f'</div>', unsafe_allow_html=True)
        
        # Display disease information
        st.markdown(f'<div class="info-box">'
                    f'<h3>About {disease_info[disease_type]["name"]}</h3>'
                    f'<p>{disease_info[disease_type]["description"]}</p>'
                    f'</div>', unsafe_allow_html=True)
        
        # Display symptoms
        st.markdown("### Common Symptoms")
        symptoms_list = ", ".join(disease_info[disease_type]["symptoms"])
        st.markdown(f"- {symptoms_list}")
        
        # Treatment, Prevention, and Cure side by side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="treatment-box">'
                        '<h4>Treatment Options</h4>'
                        '<ul>', unsafe_allow_html=True)
            for treatment in disease_info[disease_type]["treatments"]:
                st.markdown(f'<li>{treatment}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="prevention-box">'
                        '<h4>Prevention Measures</h4>'
                        '<ul>', unsafe_allow_html=True)
            for prevention in disease_info[disease_type]["prevention"]:
                st.markdown(f'<li>{prevention}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="cure-box">'
                        '<h4>Cure Status</h4>', unsafe_allow_html=True)
            st.markdown(f'<p>{disease_info[disease_type]["cure"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.markdown(f'<div class="result-box negative-result">'
                    f'<h3>✅ Low Risk of {disease_info[disease_type]["name"]} Detected</h3>'
                    f'<p>The model predicts a <strong>{probability:.1%}</strong> probability of {disease_info[disease_type]["name"]}.</p>'
                    f'<p>Always maintain a healthy lifestyle for continued wellbeing.</p>'
                    f'</div>', unsafe_allow_html=True)
        
        # Display prevention tips and cure information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prevention-box">'
                        '<h4>Prevention Tips</h4>'
                        '<ul>', unsafe_allow_html=True)
            for prevention in disease_info[disease_type]["prevention"]:
                st.markdown(f'<li>{prevention}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="cure-box">'
                        '<h4>Cure Status</h4>', unsafe_allow_html=True)
            st.markdown(f'<p>{disease_info[disease_type]["cure"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- Navigation Functions ---
# In the home_page function, modify the Quick Navigation buttons to include unique keys:
def home_page():
    """Display the home page with app overview and navigation."""
    st.markdown('<h1 class="main-header">Health Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    <div class="info-box">
        <p>Welcome to the Health Disease Prediction System! This application uses machine learning to help assess your risk 
        for common health conditions based on your health metrics.</p>
        <p><strong>Disclaimer:</strong> This tool is for educational purposes only and is not a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult with a healthcare professional regarding any health concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation cards
    st.markdown('<h2 class="sub-header">Choose a Disease to Predict</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Diabetes Card
        st.markdown("""
        <div class="disease-card">
            <div class="disease-card-title">Diabetes</div>
            <div class="disease-card-desc">Predict your diabetes risk based on health metrics like glucose levels, BMI, and more.</div>
            <div class="disease-card-stat">Affects 1 in 10 adults worldwide</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Heart Disease Card
        st.markdown("""
        <div class="disease-card" style="margin-top: 20px; border-left-color: #e74c3c;">
            <div class="disease-card-title">Heart Disease</div>
            <div class="disease-card-desc">Assess your heart health risk using metrics like blood pressure, cholesterol, and more.</div>
            <div class="disease-card-stat">Leading cause of death globally</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Liver Disease Card
        st.markdown("""
        <div class="disease-card" style="border-left-color: #f39c12;">
            <div class="disease-card-title">Liver Disease</div>
            <div class="disease-card-desc">Check your liver health risk with metrics like bilirubin levels, proteins, and enzymes.</div>
            <div class="disease-card-stat">Affects millions globally</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Kidney Disease Card
        st.markdown("""
        <div class="disease-card" style="margin-top: 20px; border-left-color: #27ae60;">
            <div class="disease-card-title">Kidney Disease</div>
            <div class="disease-card-desc">Evaluate your kidney health with metrics like blood urea, serum creatinine, and more.</div>
            <div class="disease-card-stat">850 million cases worldwide</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Direct navigation buttons
    st.markdown('<h3 class="tab-subheader" style="margin-top: 30px;">Quick Navigation</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4,col5 = st.columns(5)
    
    with col1:
        if st.button("Diabetes Prediction", key="home_diabetes_btn"):
            st.session_state.page = "diabetes"
            st.rerun()
    
    with col2:
        if st.button("Heart Disease Prediction", key="home_heart_btn"):
            st.session_state.page = "heart"
            st.rerun()
    
    with col3:
        if st.button("Liver Disease Prediction", key="home_liver_btn"):
            st.session_state.page = "liver"
            st.rerun()
    
    with col4:
        if st.button("Kidney Disease Prediction", key="home_kidney_btn"):
            st.session_state.page = "kidney"
            st.rerun()
    with col5:
        if st.button("Parkinson's Disease Prediction", key="home_parkinsons_btn"):
            st.session_state.page = "parkinsons"
            st.rerun()
# In the main function, modify the sidebar navigation buttons to include unique keys:
def main():
    # Load trained models
    models = load_models()
    
    # Set up sidebar
    st.sidebar.markdown('<h3 style="text-align: center;">Navigation</h3>', unsafe_allow_html=True)
    
    # Define the pages
    pages = {
        'home': 'Home',
        'diabetes': 'Diabetes Prediction',
        'heart': 'Heart Disease Prediction',
        'liver': 'Liver Disease Prediction',
        'kidney': 'Kidney Disease Prediction',
        'parkinsons': "Parkinson's Disease Prediction",
        'profile': 'My Health Profile',
        'about': 'About'
    }
    
    # Sidebar navigation with unique keys
    for page_id, page_name in pages.items():
        if st.sidebar.button(page_name, key=f"sidebar_{page_id}_btn"):
            st.session_state.page = page_id
            st.experimental_rerun()
    
    # Set default page
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Display the appropriate page
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'diabetes':
        diabetes_prediction_page(models)
    elif st.session_state.page == 'heart':
        heart_disease_prediction_page(models)
    elif st.session_state.page == 'liver':
        liver_disease_prediction_page(models)
    elif st.session_state.page == 'kidney':
        kidney_disease_prediction_page(models)
    elif st.session_state.page == 'parkinsons':
        parkinsons_disease_prediction_page(models)
    elif st.session_state.page == 'profile':
        user_profile_page()
    elif st.session_state.page == 'about':
        about_page()
    
    # Footer
    st.markdown('<div class="footer">© 2025 Health Disease Prediction System | Disclaimer: This application is for educational purposes only.</div>', unsafe_allow_html=True)
def diabetes_prediction_page(models):
    """Display diabetes prediction form and results."""
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Information about diabetes
    with st.expander("About Diabetes"):
        st.markdown(f"""
        <div class="info-box">
            <p>{disease_info['diabetes']['description']}</p>
            <h4>Risk Factors:</h4>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in disease_info['diabetes']['risk_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'diabetes' not in models:
        st.error("⚠️ Diabetes prediction model not loaded. Please check that the model file exists.")
        return
    
    # Input form
    st.markdown('<h2 class="sub-header">Enter Your Health Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
    
    with col2:
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
        age = st.number_input('Age', min_value=0, max_value=120, value=33, step=1)
    
    # Prediction button
    if st.button('Predict Diabetes Risk'):
        with st.spinner('Analyzing your data...'):
            # Simulate delay for better UX
            time.sleep(1)
            # Make prediction
            prediction, probability = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                                                       insulin, bmi, dpf, age, models['diabetes'])
            # Display results
            display_results(prediction, probability, 'diabetes')

def heart_disease_prediction_page(models):
    """Display heart disease prediction form and results."""
    st.markdown('<h1 class="main-header">Heart Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Information about heart disease
    with st.expander("About Heart Disease"):
        st.markdown(f"""
        <div class="info-box">
            <p>{disease_info['heart']['description']}</p>
            <h4>Risk Factors:</h4>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in disease_info['heart']['risk_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'heart' not in models:
        st.error("⚠️ Heart disease prediction model not loaded. Please check that the model file exists.")
        return
    
    # Input form
    st.markdown('<h2 class="sub-header">Enter Your Health Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox('Sex', [('Male', 1), ('Female', 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox('Chest Pain Type', 
                         [('Typical Angina', 0), ('Atypical Angina', 1), 
                          ('Non-anginal Pain', 2), ('Asymptomatic', 3)], 
                         format_func=lambda x: x[0])[1]
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=300, value=120, step=1)
        chol = st.number_input('Cholesterol (mg/dl)', min_value=50, max_value=600, value=200, step=1)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox('Resting ECG Results', 
                              [('Normal', 0), ('ST-T Wave Abnormality', 1), ('Left Ventricular Hypertrophy', 2)], 
                              format_func=lambda x: x[0])[1]
    
    with col2:
        thalach = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150, step=1)
        exang = st.selectbox('Exercise Induced Angina', [('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                            [('Upsloping', 0), ('Flat', 1), ('Downsloping', 2)], 
                            format_func=lambda x: x[0])[1]
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0, step=1)
        thal = st.selectbox('Thalassemia', 
                           [('Normal', 0), ('Fixed Defect', 1), ('Reversible Defect', 2)], 
                           format_func=lambda x: x[0])[1]
    
    # Prediction button
    if st.button('Predict Heart Disease Risk'):
        with st.spinner('Analyzing your data...'):
            # Simulate delay for better UX
            time.sleep(1)
            # Make prediction
            prediction, probability = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                                                            thalach, exang, oldpeak, slope, ca, thal, 
                                                            models['heart'])
            # Display results
            display_results(prediction, probability, 'heart')

def liver_disease_prediction_page(models):
    """Display liver disease prediction form and results."""
    st.markdown('<h1 class="main-header">Liver Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Information about liver disease
    with st.expander("About Liver Disease"):
        st.markdown(f"""
        <div class="info-box">
            <p>{disease_info['liver']['description']}</p>
            <h4>Risk Factors:</h4>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in disease_info['liver']['risk_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'liver' not in models:
        st.error("⚠️ Liver disease prediction model not loaded. Please check that the model file exists.")
        return
    
    # Input form
    st.markdown('<h2 class="sub-header">Enter Your Health Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=45, step=1, key='liver_age')
        gender = st.selectbox('Gender', [('Male', 1), ('Female', 0)], format_func=lambda x: x[0])[1]
        total_bilirubin = st.number_input('Total Bilirubin (mg/dL)', min_value=0.0, max_value=30.0, value=1.0, step=0.1, format="%.1f")
        direct_bilirubin = st.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, max_value=20.0, value=0.3, step=0.1, format="%.1f")
        alkaline_phosphatase = st.number_input('Alkaline Phosphatase (IU/L)', min_value=20, max_value=2000, value=290, step=1)
    
    with col2:
        sgpt = st.number_input('SGPT (IU/L)', min_value=1, max_value=2000, value=40, step=1)
        sgot = st.number_input('SGOT (IU/L)', min_value=1, max_value=2000, value=40, step=1)
        total_proteins = st.number_input('Total Proteins (g/dL)', min_value=1.0, max_value=15.0, value=6.8, step=0.1, format="%.1f")
        albumin = st.number_input('Albumin (g/dL)', min_value=0.5, max_value=10.0, value=3.5, step=0.1, format="%.1f")
        ag_ratio = st.number_input('A/G Ratio', min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    
    # Prediction button
    if st.button('Predict Liver Disease Risk'):
        with st.spinner('Analyzing your data...'):
            # Simulate delay for better UX
            time.sleep(1)
            # Make prediction
            prediction, probability = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, 
                                                           alkaline_phosphatase, sgpt, sgot, total_proteins, 
                                                           albumin, ag_ratio, models['liver'])
            # Display results
            display_results(prediction, probability, 'liver')
            
def parkinsons_disease_prediction_page(models):
    """Display Parkinson's disease prediction form and results."""
    st.markdown('<h1 class="main-header">Parkinson\'s Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Info section
    with st.expander("About Parkinson's Disease"):
        st.markdown("""
        <div class="info-box">
            <p>Parkinson's disease is a progressive disorder of the nervous system that affects movement. It develops gradually, sometimes starting with a barely noticeable tremor.</p>
            <h4>Common Symptoms:</h4>
            <ul>
                <li>Tremors</li>
                <li>Slowed movement (bradykinesia)</li>
                <li>Rigid muscles</li>
                <li>Speech or writing changes</li>
                <li>Impaired posture and balance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Check model
    if 'parkinsons' not in models:
        st.error("⚠️ Parkinson's prediction model not loaded. Please check that the model file exists.")
        return

    st.markdown('<h2 class="sub-header">Enter Your Health Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", value=150.0)
        flo = st.number_input("MDVP:Flo(Hz)", value=100.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.005)
        shimmer = st.number_input("MDVP:Shimmer", value=0.03)
        rap = st.number_input("MDVP:RAP", value=0.003)
        ppq = st.number_input("MDVP:PPQ", value=0.004)
        ddp = st.number_input("Jitter:DDP", value=0.009)
        shimmer_db = st.number_input("Shimmer:dB", value=0.03)
        apq3 = st.number_input("Shimmer:APQ3", value=0.01)

    with col2:
        apq5 = st.number_input("Shimmer:APQ5", value=0.02)
        apq = st.number_input("Shimmer:APQ", value=0.02)
        dda = st.number_input("Shimmer:DDA", value=0.01)
        nhr = st.number_input("NHR", value=0.01)
        hnr = st.number_input("HNR", value=20.0)
        rpde = st.number_input("RPDE", value=0.5)
        dfa = st.number_input("DFA", value=0.6)
        spread1 = st.number_input("Spread1", value=-5.0)
        spread2 = st.number_input("Spread2", value=0.1)
        d2 = st.number_input("D2", value=2.0)
        ppe = st.number_input("PPE", value=0.1)

    if st.button("Predict Parkinson's Risk"):
        with st.spinner('Analyzing your data...'):
            time.sleep(1)
            prediction, probability = predict_parkinsons_disease(
                fo, fhi, flo, jitter_percent, shimmer, rap, ppq, ddp, shimmer_db,
                apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe,
                models['parkinsons']
            )
            display_results(prediction, probability, 'parkinsons')


def kidney_disease_prediction_page(models):
    """Display kidney disease prediction form and results."""
    st.markdown('<h1 class="main-header">Kidney Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Information about kidney disease
    with st.expander("About Kidney Disease"):
        st.markdown(f"""
        <div class="info-box">
            <p>{disease_info['kidney']['description']}</p>
            <h4>Risk Factors:</h4>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in disease_info['kidney']['risk_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'kidney' not in models:
        st.error("⚠️ Kidney disease prediction model not loaded. Please check that the model file exists.")
        return
    
    # Input form
    st.markdown('<h2 class="sub-header">Enter Your Health Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=45, step=1, key='kidney_age')
        bp = st.number_input('Blood Pressure (mm Hg)', min_value=50, max_value=200, value=80, step=1)
        sg = st.selectbox('Specific Gravity', 
                         [('1.005', 0), ('1.010', 1), ('1.015', 2), ('1.020', 3), ('1.025', 4)], 
                         index=2, format_func=lambda x: x[0])[1]
        albumin = st.selectbox('Albumin', 
                              [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5)], 
                              format_func=lambda x: x[0])[1]
        sugar = st.selectbox('Sugar', 
                             [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5)], 
                             format_func=lambda x: x[0])[1]
    
    with col2:
        rbc = st.selectbox('Red Blood Cells', [('Normal', 0), ('Abnormal', 1)], format_func=lambda x: x[0])[1]
        pc = st.selectbox('Pus Cell', [('Normal', 0), ('Abnormal', 1)], format_func=lambda x: x[0])[1]
        pcc = st.selectbox('Pus Cell Clumps', [('Not Present', 0), ('Present', 1)], format_func=lambda x: x[0])[1]
        bu = st.number_input('Blood Urea (mg/dL)', min_value=1.0, max_value=200.0, value=44.0, step=0.1, format="%.1f")
        sc = st.number_input('Serum Creatinine (mg/dL)', min_value=0.1, max_value=15.0, value=1.3, step=0.1, format="%.1f")
        sod = st.number_input('Sodium (mEq/L)', min_value=100, max_value=200, value=135, step=1)
    
    # Prediction button
    if st.button('Predict Kidney Disease Risk'):
        with st.spinner('Analyzing your data...'):
            # Simulate delay for better UX
            time.sleep(1)
            # Make prediction
            prediction, probability = predict_kidney_disease(age, bp, sg, albumin, sugar, rbc, pc, pcc, bu, sc, sod, 
                                                           models['kidney'])
            # Display results
            display_results(prediction, probability, 'kidney')

def user_profile_page():
    """Display and manage user health profile."""
    st.markdown('<h1 class="main-header">My Health Profile</h1>', unsafe_allow_html=True)
    
    # Profile form
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<h3>Personal Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", value="John Doe")
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
    
    with col2:
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
        email = st.text_input("Email", value="john.doe@example.com")
        phone = st.text_input("Phone Number", value="+1234567890")
        emergency_contact = st.text_input("Emergency Contact", value="Jane Doe: +0987654321")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Medical history
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<h3>Medical History</h3>', unsafe_allow_html=True)
    
    existing_conditions = st.multiselect(
        "Existing Medical Conditions",
        ["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease", "Liver Disease", "Asthma", "Cancer", "Other"],
        default=[]
    )
    
    allergies = st.text_area("Allergies", value="None")
    
    medications = st.text_area("Current Medications", value="None")
    
    family_history = st.multiselect(
        "Family Medical History",
        ["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease", "Liver Disease", "Cancer", "Other"],
        default=[]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lifestyle information
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<h3>Lifestyle Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        exercise = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
        sleep = st.number_input("Average Sleep Hours", min_value=1, max_value=24, value=7)
        smoker = st.checkbox("Smoker")
    
    with col2:
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Heavy"])
        diet = st.selectbox("Diet Type", ["Regular", "Vegetarian", "Vegan", "Keto", "Paleo", "Other"])
        stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save profile button
    if st.button("Save Profile"):
        st.success("Profile saved successfully!")
        st.info("Note: In a production environment, this data would be securely stored in a database.")

def about_page():
    """Display information about the app and the diseases it can predict."""
    st.markdown('<h1 class="main-header">About This App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>The Health Disease Prediction System is an AI-powered application that helps users assess their risk for various 
        common health conditions. By providing your health metrics, the system uses machine learning models to estimate 
        your risk level and provides relevant health information.</p>
        
       
    </div>
    """, unsafe_allow_html=True)
    
    # Disease information
    st.markdown('<h2 class="sub-header">Diseases Covered</h2>', unsafe_allow_html=True)
    
    for disease_key, disease_data in disease_info.items():
        with st.expander(f"{disease_data['name']}"):
            st.markdown(f"""
            <h4>{disease_data['name']}</h4>
            <p>{disease_data['description']}</p>
            
            <h5>Common Symptoms</h5>
            <ul>
                {''.join([f'<li>{symptom}</li>' for symptom in disease_data['symptoms']])}
            </ul>
            
            <h5>Risk Factors</h5>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in disease_data['risk_factors']])}
            </ul>
            """, unsafe_allow_html=True)
    
    # Technical details
    st.markdown('<h2 class="sub-header">Technical Details</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>This application is built using:</p>
        <ul>
            <li><strong>Streamlit</strong> - For the web application framework</li>
            <li><strong>Scikit-learn</strong> - For machine learning models</li>
            <li><strong>Pandas & NumPy</strong> - For data handling</li>
            <li><strong>Plotly</strong> - For interactive visualizations</li>
        </ul>
        <p>The prediction models are trained on standard medical datasets and are regularly updated to improve accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
  

# --- Main app ---
def main():
    # Load trained models
    models = load_models()
    
    # Set up sidebar
    st.sidebar.markdown('<h3 style="text-align: center;">Navigation</h3>', unsafe_allow_html=True)
    
    # Define the pages
    pages = {
        'home': 'Home',
        'diabetes': 'Diabetes Prediction',
        'heart': 'Heart Disease Prediction',
        'liver': 'Liver Disease Prediction',
        'kidney': 'Kidney Disease Prediction',
        'parkinsons': 'Parkinsons Disease Prediction',
        'profile': 'My Health Profile',
        'about': 'About'
    }
    # Parkinson's model

    # Sidebar navigation
    for page_id, page_name in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_id
            st.rerun()
    
    # Set default page
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Display the appropriate page
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'diabetes':
        diabetes_prediction_page(models)
    elif st.session_state.page == 'heart':
        heart_disease_prediction_page(models)
    elif st.session_state.page == 'liver':
        liver_disease_prediction_page(models)
    elif st.session_state.page == 'kidney':
        kidney_disease_prediction_page(models)
    elif st.session_state.page == 'parkinsons':
        parkinsons_disease_prediction_page(models)
    elif st.session_state.page == 'profile':
        user_profile_page()
    elif st.session_state.page == 'about':
        about_page()
        
if __name__ == '__main__':
    main()