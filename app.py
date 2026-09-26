import os
from dotenv import load_dotenv
load_dotenv()
import math
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from flask_cors import CORS
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

history_data = []
appointments_data = []


VALID_USER = os.environ.get("APP_USER", "admin")
VALID_PASS = os.environ.get("APP_PASS", "admin123")

FEATURE_NAMES = ["radius", "texture", "perimeter", "area"]
THRESHOLDS = {"radius": 15.0, "texture": 20.0, "perimeter": 100.0, "area": 700.0}


def validate_features(form):
    features = []
    for name in FEATURE_NAMES:
        raw = form.get(name, "").strip()
        try:
            val = float(raw)
            if math.isnan(val) or math.isinf(val):
                raise ValueError
            if val < 0:
                raise ValueError
        except (ValueError, TypeError):
            return None, f"Invalid value for {name}. Please enter a positive number."
        features.append(val)
    return features, None


def recommend_doctor(pred):
    if pred == 1:
        return {
            "type": "Oncologist",
            "name": "Dr. Rajiv Sharma",
            "hospital": "AIIMS Delhi",
            "contact": "+91 98765 43210",
            "advice": "Please book an appointment immediately and consider a biopsy."
        }
    return {
        "type": "General Physician",
        "name": "Dr. Neha Verma",
        "hospital": "City Care Clinic",
        "contact": "+91 91234 56780",
        "advice": "Continue regular monitoring every 6 months."
    }


def clinical_risk_analysis(features):
    factors = []
    score = 0
    for name, val in zip(FEATURE_NAMES, features):
        thresh = THRESHOLDS[name]
        if val > thresh:
            factors.append(f"{name.capitalize()} is high ({val:.1f} > {thresh})")
            score += 25
    score = min(score, 100)
    if score >= 75:
        level = "High"
    elif score >= 50:
        level = "Medium"
    else:
        level = "Low"
    return score, level, factors if factors else ["All values are within normal range"]

def get_clinical_recommendations(pred, risk_level, features):
    """Generate clinical recommendations based on prediction and risk level"""
    recommendations = {
        "treatment": [],
        "lifestyle": [],
        "followup": [],
        "surgery": None,
        "medication": []
    }
    
    if pred == 1:  # Malignant
        recommendations["treatment"] = [
            "🔬 Immediate biopsy to confirm diagnosis and determine cancer type",
            "🏥 Consultation with oncology team for treatment planning",
            "📊 Additional imaging tests (MRI, CT scan, PET scan) to assess spread"
        ]
        
        if risk_level == "High":
            recommendations["surgery"] = {
                "type": "Mastectomy or Lumpectomy",
                "description": "Surgical removal of cancerous tissue. Type depends on tumor size and location.",
                "urgency": "High Priority - Schedule within 2-4 weeks"
            }
            recommendations["medication"] = [
                "Chemotherapy (Doxorubicin, Cyclophosphamide)",
                "Targeted therapy (Herceptin for HER2-positive)",
                "Hormone therapy (Tamoxifen for hormone-receptor positive)"
            ]
        else:
            recommendations["surgery"] = {
                "type": "Lumpectomy (Breast-conserving surgery)",
                "description": "Removal of tumor and small margin of surrounding tissue.",
                "urgency": "Schedule within 4-6 weeks"
            }
            recommendations["medication"] = [
                "Radiation therapy post-surgery",
                "Hormone therapy if receptor-positive"
            ]
        
        recommendations["followup"] = [
            "Weekly monitoring during treatment",
            "Post-treatment scans every 3 months for first year",
            "Genetic counseling for BRCA1/BRCA2 testing"
        ]
        
        recommendations["lifestyle"] = [
            "🥗 Anti-inflammatory diet rich in fruits and vegetables",
            "🚭 Avoid smoking and alcohol consumption",
            "💪 Light exercise as tolerated during treatment",
            "😴 Adequate rest and stress management"
        ]
        
    else:  # Benign
        if risk_level == "Medium" or risk_level == "High":
            recommendations["treatment"] = [
                "🔍 Fine needle aspiration or core biopsy to rule out malignancy",
                "📸 Mammogram and ultrasound every 6 months",
                "👨⚕️ Regular monitoring by breast specialist"
            ]
            recommendations["followup"] = [
                "Clinical examination every 6 months",
                "Annual mammogram and ultrasound",
                "Self-examination monthly"
            ]
        else:
            recommendations["treatment"] = [
                "✅ Continue routine screening",
                "📅 Annual mammogram after age 40",
                "🔍 Self-examination monthly"
            ]
            recommendations["followup"] = [
                "Annual clinical breast examination",
                "Mammogram every 1-2 years based on age"
            ]
        
        recommendations["lifestyle"] = [
            "🥗 Maintain healthy weight (BMI 18.5-24.9)",
            "🏃 Regular exercise (150 min/week moderate activity)",
            "🚭 Avoid smoking and limit alcohol",
            "🥦 Diet rich in fiber, low in processed foods"
        ]
        
        recommendations["medication"] = ["No medication required at this time"]
    
    return recommendations

def get_dietary_recommendations(pred, risk_level):
    """Generate dietary recommendations based on diagnosis"""
    
    if pred == 1:  # Malignant
        foods_to_avoid = [
            {"icon": "🍔", "name": "Processed Meats", "reason": "High in nitrates, linked to cancer growth"},
            {"icon": "🍰", "name": "Refined Sugar", "reason": "Feeds cancer cells, causes inflammation"},
            {"icon": "🍺", "name": "Alcohol", "reason": "Increases estrogen levels, impairs immunity"},
            {"icon": "🍟", "name": "Fried Foods", "reason": "Contains trans fats, promotes inflammation"},
            {"icon": "🥤", "name": "Sugary Drinks", "reason": "Spikes blood sugar, weakens immune system"},
            {"icon": "🧂", "name": "Excess Salt", "reason": "May interfere with treatment, causes bloating"},
            {"icon": "🍕", "name": "Fast Food", "reason": "High in unhealthy fats and preservatives"},
            {"icon": "🥓", "name": "Red Meat", "reason": "Limit to once per week, choose lean cuts"}
        ]
        
        foods_to_include = [
            {"icon": "🥦", "name": "Cruciferous Vegetables", "reason": "Broccoli, cauliflower - anti-cancer compounds"},
            {"icon": "🫐", "name": "Berries", "reason": "Rich in antioxidants, fights free radicals"},
            {"icon": "🐟", "name": "Fatty Fish", "reason": "Omega-3 reduces inflammation"},
            {"icon": "🥬", "name": "Leafy Greens", "reason": "Spinach, kale - packed with vitamins"},
            {"icon": "🥜", "name": "Nuts & Seeds", "reason": "Healthy fats, protein, selenium"},
            {"icon": "🫘", "name": "Legumes", "reason": "Beans, lentils - fiber and protein"},
            {"icon": "🍊", "name": "Citrus Fruits", "reason": "Vitamin C boosts immunity"},
            {"icon": "🧄", "name": "Garlic & Onions", "reason": "Natural anti-cancer properties"}
        ]
        
        meal_plan = [
            {
                "time": "7:00 AM - Breakfast",
                "title": "Anti-Inflammatory Start",
                "meal_items": [  # Changed from "items"
                    "Oatmeal with blueberries and walnuts",
                    "Green tea (antioxidant-rich)",
                    "1 boiled egg or Greek yogurt"
                ]
            },
            {
                "time": "10:00 AM - Snack",
                "title": "Energy Boost",
                "meal_items": [  # Changed from "items"
                    "Apple slices with almond butter",
                    "Handful of mixed nuts"
                ]
            },
            {
                "time": "1:00 PM - Lunch",
                "title": "Nutrient-Dense Meal",
                "meal_items": [  # Changed from "items"
                    "Grilled salmon with quinoa",
                    "Large mixed green salad with olive oil",
                    "Steamed broccoli and carrots"
                ]
            },
            {
                "time": "4:00 PM - Snack",
                "title": "Afternoon Fuel",
                "meal_items": [  # Changed from "items"
                    "Carrot and celery sticks with hummus",
                    "Herbal tea"
                ]
            },
            {
                "time": "7:00 PM - Dinner",
                "title": "Light & Healing",
                "meal_items": [  # Changed from "items"
                    "Grilled chicken breast or tofu",
                    "Sweet potato and Brussels sprouts",
                    "Lentil soup"
                ]
            }
        ]

        
    else:  # Benign
        foods_to_avoid = [
            {"icon": "🍰", "name": "Excess Sugar", "reason": "Maintain healthy weight"},
            {"icon": "🍺", "name": "Alcohol", "reason": "Limit to 1 drink per day or avoid"},
            {"icon": "🍟", "name": "Trans Fats", "reason": "Increases inflammation"},
            {"icon": "🥤", "name": "Soda", "reason": "Empty calories, no nutrition"},
            {"icon": "🧂", "name": "High Sodium", "reason": "May cause water retention"},
            {"icon": "🍕", "name": "Processed Foods", "reason": "Low nutritional value"}
        ]
        
        foods_to_include = [
            {"icon": "🥦", "name": "Vegetables", "reason": "5+ servings daily for prevention"},
            {"icon": "🍎", "name": "Fresh Fruits", "reason": "Natural vitamins and fiber"},
            {"icon": "🌾", "name": "Whole Grains", "reason": "Brown rice, quinoa, oats"},
            {"icon": "🐟", "name": "Lean Protein", "reason": "Fish, chicken, plant-based"},
            {"icon": "🥜", "name": "Healthy Fats", "reason": "Avocado, nuts, olive oil"},
            {"icon": "🫘", "name": "Fiber-Rich Foods", "reason": "Supports digestive health"}
        ]
        
        meal_plan = [
            {
                "time": "7:30 AM - Breakfast",
                "title": "Balanced Start",
                "items": [
                    "Whole grain toast with avocado",
                    "Scrambled eggs or tofu",
                    "Fresh fruit smoothie"
                ]
            },
            {
                "time": "10:30 AM - Snack",
                "title": "Mid-Morning",
                "items": [
                    "Greek yogurt with berries",
                    "Handful of almonds"
                ]
            },
            {
                "time": "1:00 PM - Lunch",
                "title": "Wholesome Meal",
                "items": [
                    "Grilled chicken salad with mixed greens",
                    "Whole grain roll",
                    "Fresh fruit"
                ]
            },
            {
                "time": "4:00 PM - Snack",
                "title": "Afternoon Pick-Me-Up",
                "items": [
                    "Hummus with veggie sticks",
                    "Green tea"
                ]
            },
            {
                "time": "7:00 PM - Dinner",
                "title": "Nutritious Evening",
                "items": [
                    "Baked fish with herbs",
                    "Roasted vegetables",
                    "Brown rice or quinoa"
                ]
            }
        ]
    
    # Common allergies
    allergies = [
        {
            "icon": "🥜",
            "name": "Nuts & Peanuts",
            "description": "Common allergen, especially during treatment when immune system is compromised",
            "symptoms": "Hives, swelling, difficulty breathing, digestive issues"
        },
        {
            "icon": "🥛",
            "name": "Dairy Products",
            "description": "May cause digestive issues during chemotherapy",
            "symptoms": "Bloating, gas, diarrhea, nausea"
        },
        {
            "icon": "🌾",
            "name": "Gluten",
            "description": "Some patients develop sensitivity during treatment",
            "symptoms": "Abdominal pain, fatigue, brain fog"
        },
        {
            "icon": "🦐",
            "name": "Shellfish",
            "description": "High-risk allergen, avoid if uncertain",
            "symptoms": "Severe reactions possible, seek immediate help"
        }
    ]
    
    # Supplements
    supplements = [
        {"icon": "🟡", "name": "Vitamin D3", "description": "2000 IU daily - supports bone health"},
        {"icon": "🔴", "name": "Vitamin C", "description": "1000mg daily - boosts immunity"},
        {"icon": "🟠", "name": "Omega-3", "description": "Fish oil - reduces inflammation"},
        {"icon": "🟢", "name": "Probiotics", "description": "Supports gut health during treatment"}
    ]
    
    hydration = {
        "water": "8-10 glasses (64-80 oz) daily. Increase during chemotherapy."
    }
    
    return {
        "foods_to_avoid": foods_to_avoid,
        "foods_to_include": foods_to_include,
        "meal_plan": meal_plan,
        "allergies": allergies,
        "supplements": supplements,
        "hydration": hydration
    }

def generate_prescription(pred, risk_level, features, patient_info=None):
    """Generate medication dosage prescription based on diagnosis"""
    
    prescription = {
        "patient_info": patient_info or {
            "name": "Patient Name",
            "age": "N/A",
            "gender": "N/A",
            "date": datetime.now().strftime("%B %d, %Y")
        },
        "diagnosis": "Malignant Breast Cancer" if pred == 1 else "Benign Breast Condition",
        "risk_level": risk_level,
        "medications": [],
        "instructions": [],
        "follow_up": "",
        "warnings": []
    }
    
    if pred == 1:  # Malignant
        if risk_level == "High":
            prescription["medications"] = [
                {
                    "name": "Doxorubicin (Adriamycin)",
                    "dosage": "60 mg/m² IV",
                    "frequency": "Every 21 days for 4 cycles",
                    "duration": "12 weeks",
                    "purpose": "Chemotherapy - kills cancer cells"
                },
                {
                    "name": "Cyclophosphamide (Cytoxan)",
                    "dosage": "600 mg/m² IV",
                    "frequency": "Every 21 days for 4 cycles",
                    "duration": "12 weeks",
                    "purpose": "Chemotherapy - prevents cell division"
                },
                {
                    "name": "Tamoxifen",
                    "dosage": "20 mg oral",
                    "frequency": "Once daily",
                    "duration": "5 years",
                    "purpose": "Hormone therapy - blocks estrogen"
                },
                {
                    "name": "Ondansetron (Zofran)",
                    "dosage": "8 mg oral",
                    "frequency": "Every 8 hours as needed",
                    "duration": "During chemotherapy",
                    "purpose": "Anti-nausea medication"
                },
                {
                    "name": "Filgrastim (Neupogen)",
                    "dosage": "5 mcg/kg subcutaneous",
                    "frequency": "Daily for 7-10 days",
                    "duration": "After each chemo cycle",
                    "purpose": "Boosts white blood cell count"
                }
            ]
            
            prescription["instructions"] = [
                "Take all medications exactly as prescribed",
                "Do not skip chemotherapy appointments",
                "Report fever above 100.4°F immediately",
                "Stay hydrated - drink 8-10 glasses of water daily",
                "Avoid crowds and sick people during treatment",
                "Use contraception during treatment",
                "Take anti-nausea medication 30 minutes before meals"
            ]
            
            prescription["follow_up"] = "Weekly blood tests during chemotherapy, oncology visit every 3 weeks"
            
            prescription["warnings"] = [
                "⚠️ May cause severe nausea, vomiting, and hair loss",
                "⚠️ Increased risk of infection - monitor temperature daily",
                "⚠️ May cause fatigue and weakness",
                "⚠️ Avoid pregnancy during treatment",
                "⚠️ Report unusual bleeding or bruising immediately"
            ]
            
        else:  # Medium/Low risk malignant
            prescription["medications"] = [
                {
                    "name": "Tamoxifen",
                    "dosage": "20 mg oral",
                    "frequency": "Once daily",
                    "duration": "5 years",
                    "purpose": "Hormone therapy - reduces recurrence risk"
                },
                {
                    "name": "Anastrozole (Arimidex)",
                    "dosage": "1 mg oral",
                    "frequency": "Once daily",
                    "duration": "5 years",
                    "purpose": "Aromatase inhibitor - lowers estrogen"
                },
                {
                    "name": "Calcium + Vitamin D",
                    "dosage": "1200 mg + 800 IU",
                    "frequency": "Once daily",
                    "duration": "Ongoing",
                    "purpose": "Bone health support"
                }
            ]
            
            prescription["instructions"] = [
                "Take medications at the same time each day",
                "Do not miss doses",
                "Report any unusual symptoms to your doctor",
                "Maintain regular exercise routine",
                "Attend all follow-up appointments"
            ]
            
            prescription["follow_up"] = "Oncology visit every 3 months, mammogram every 6 months"
            
            prescription["warnings"] = [
                "⚠️ May cause hot flashes and joint pain",
                "⚠️ Slight increased risk of blood clots",
                "⚠️ Report leg swelling or chest pain immediately"
            ]
    
    else:  # Benign
        if risk_level == "Medium" or risk_level == "High":
            prescription["medications"] = [
                {
                    "name": "Ibuprofen",
                    "dosage": "400 mg oral",
                    "frequency": "Every 6-8 hours as needed",
                    "duration": "For pain management",
                    "purpose": "Pain relief and anti-inflammatory"
                },
                {
                    "name": "Vitamin E",
                    "dosage": "400 IU",
                    "frequency": "Once daily",
                    "duration": "3-6 months",
                    "purpose": "May reduce breast pain"
                },
                {
                    "name": "Evening Primrose Oil",
                    "dosage": "1000 mg",
                    "frequency": "Twice daily",
                    "duration": "3-6 months",
                    "purpose": "Natural supplement for breast health"
                }
            ]
            
            prescription["instructions"] = [
                "Monitor symptoms and report any changes",
                "Perform monthly self-examinations",
                "Maintain healthy weight and exercise regularly",
                "Limit caffeine and alcohol intake"
            ]
            
            prescription["follow_up"] = "Clinical examination every 6 months, mammogram annually"
            
            prescription["warnings"] = [
                "⚠️ Report any new lumps or changes immediately",
                "⚠️ Ibuprofen may cause stomach upset - take with food"
            ]
            
        else:  # Low risk benign
            prescription["medications"] = [
                {
                    "name": "Multivitamin",
                    "dosage": "1 tablet",
                    "frequency": "Once daily",
                    "duration": "Ongoing",
                    "purpose": "General health maintenance"
                },
                {
                    "name": "Vitamin D3",
                    "dosage": "2000 IU",
                    "frequency": "Once daily",
                    "duration": "Ongoing",
                    "purpose": "Bone and immune health"
                }
            ]
            
            prescription["instructions"] = [
                "Continue routine breast self-examinations monthly",
                "Maintain healthy lifestyle",
                "Annual mammogram after age 40",
                "Report any changes to your doctor"
            ]
            
            prescription["follow_up"] = "Annual clinical breast examination"
            
            prescription["warnings"] = [
                "⚠️ Report any new symptoms or concerns"
            ]
    
    return prescription


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_user", methods=["POST"])
def login_user():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == VALID_USER and password == VALID_PASS:
        session["user"] = username
        flash("Login successful!", "success")
        return redirect(url_for("dashboard"))
    flash("Invalid credentials!", "error")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))
    return render_template("dashboard.html", table=None, benign=0, malignant=0)


# ── PREDICT ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    # Get patient information (name, age, gender)
    patient_name = request.form.get("patient_name", "").strip()
    patient_age = request.form.get("patient_age", "").strip()
    patient_gender = request.form.get("patient_gender", "").strip()
    
    # Validate patient info
    if not patient_name or not patient_age or not patient_gender:
        flash("Please provide patient name, age, and gender.", "error")
        return redirect(url_for("home"))
    
    # Store in session
    session["patient_name"] = patient_name
    session["patient_age"] = patient_age
    session["patient_gender"] = patient_gender
    
    # Get and validate features
    features, error = validate_features(request.form)
    if error:
        flash(error, "error")
        return redirect(url_for("home"))

    # Store features in session for prescription
    session["last_radius"] = features[0]
    session["last_texture"] = features[1]
    session["last_perimeter"] = features[2]
    session["last_area"] = features[3]

    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    prob_all = model.predict_proba(scaled)[0]

    benign = round(float(prob_all[0]) * 100, 2)
    malignant = round(float(prob_all[1]) * 100, 2)
    result = "Malignant" if pred == 1 else "Benign"
    confidence = malignant if pred == 1 else benign

    doctor = recommend_doctor(pred)
    risk_score, risk_level, factors = clinical_risk_analysis(features)
    recommendations = get_clinical_recommendations(pred, risk_level, features)

    # Store in session for dietary and prescription pages
    session["last_result"] = result
    session["last_risk_level"] = risk_level

    history_data.append({
        "id": len(history_data) + 1,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "result": result,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "factors": factors,
        "doctor": doctor["name"],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    })

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        benign=benign,
        malignant=malignant,
        doctor=doctor,
        risk_score=risk_score,
        risk_level=risk_level,
        factors=factors,
        recommendations=recommendations,
        inputs=dict(zip(FEATURE_NAMES, features)),
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender
    )




@app.route("/dietary")
def dietary():
    """Dietary recommendations page with full error handling"""
    try:
        # Debug logging
        print("=" * 50)
        print("DIETARY ROUTE ACCESSED")
        print(f"Session keys: {list(session.keys())}")
        print(f"History count: {len(history_data)}")
        
        # Get result and risk level
        result = session.get("last_result")
        risk_level = session.get("last_risk_level")
        
        print(f"Session result: {result}")
        print(f"Session risk_level: {risk_level}")
        
        # Fallback to history if session is empty
        if not result and len(history_data) > 0:
            last_record = history_data[-1]
            result = last_record.get("result", "Benign")
            risk_level = last_record.get("risk_level", "Low")
            print(f"Using history - result: {result}, risk: {risk_level}")
        
        # Final fallback to defaults
        if not result:
            result = "Benign"
            risk_level = "Low"
            print("Using defaults")
        
        # Convert to prediction value
        pred = 1 if result == "Malignant" else 0
        print(f"Prediction value: {pred}")
        
        # Get dietary data
        print("Calling get_dietary_recommendations...")
        dietary_data = get_dietary_recommendations(pred, risk_level)
        print("Dietary data retrieved successfully")
        
        # Render template
        print("Rendering template...")
        return render_template(
            "dietary.html",
            result=result,
            foods_to_avoid=dietary_data.get("foods_to_avoid", []),
            foods_to_include=dietary_data.get("foods_to_include", []),
            meal_plan=dietary_data.get("meal_plan", []),
            allergies=dietary_data.get("allergies", []),
            supplements=dietary_data.get("supplements", []),
            hydration=dietary_data.get("hydration", {})
        )
        
    except Exception as e:
        print("=" * 50)
        print("ERROR IN DIETARY ROUTE:")
        print(str(e))
        import traceback
        traceback.print_exc()
        print("=" * 50)
        
        # Return a simple error page instead of crashing
        return f"""
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; padding: 50px; background: #1a1a2e; color: white;">
            <h1>⚠️ Error Loading Dietary Guide</h1>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please <a href="/" style="color: #00c9a7;">go back to home</a> and make a prediction first.</p>
            <hr>
            <pre style="background: #2a2a3d; padding: 20px; border-radius: 10px; overflow: auto;">
{traceback.format_exc()}
            </pre>
        </body>
        </html>
        """

@app.route("/prescription")
def prescription():
    """Prescription page"""
    try:
        # Get data from session
        result = session.get("last_result", "Benign")
        risk_level = session.get("last_risk_level", "Low")
        
        # Get from history if session is empty
        if not result and history_data:
            last_record = history_data[-1]
            result = last_record.get("result", "Benign")
            risk_level = last_record.get("risk_level", "Low")
        
        pred = 1 if result == "Malignant" else 0
        
        # Get patient info from session if available
        patient_info = {
            "name": session.get("patient_name", "Patient Name"),
            "age": session.get("patient_age", "N/A"),
            "gender": session.get("patient_gender", "N/A"),
            "date": datetime.now().strftime("%B %d, %Y"),
            "prescription_id": f"RX-{datetime.now().strftime('%Y%m%d')}-{len(history_data) + 1:04d}"
        }
        
        # Get features from session
        features = [
            session.get("last_radius", 0),
            session.get("last_texture", 0),
            session.get("last_perimeter", 0),
            session.get("last_area", 0)
        ]
        
        prescription_data = generate_prescription(pred, risk_level, features, patient_info)
        
        return render_template(
            "prescription.html",
            **prescription_data
        )
        
    except Exception as e:
        print(f"Error in prescription route: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error loading prescription: {str(e)}", "error")
        return redirect(url_for("home"))


@app.route("/download-prescription")
def download_prescription():
    """Generate and download prescription PDF"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import Spacer
        
        # Get data from session
        result = session.get("last_result", "Benign")
        risk_level = session.get("last_risk_level", "Low")
        pred = 1 if result == "Malignant" else 0
        
        patient_info = {
            "name": session.get("patient_name", "Patient Name"),
            "age": session.get("patient_age", "N/A"),
            "gender": session.get("patient_gender", "N/A"),
            "date": datetime.now().strftime("%B %d, %Y"),
            "prescription_id": f"RX-{datetime.now().strftime('%Y%m%d')}-{len(history_data) + 1:04d}"
        }
        
        features = [0, 0, 0, 0]
        prescription_data = generate_prescription(pred, risk_level, features, patient_info)
        
        # Create PDF
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "prescription.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        header_style = styles["Heading1"]
        header_style.textColor = colors.HexColor("#00c9a7")
        story.append(Paragraph("🏥 MEDICAL PRESCRIPTION", header_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Info
        story.append(Paragraph(f"<b>Prescription ID:</b> {patient_info['prescription_id']}", styles["Normal"]))
        story.append(Paragraph(f"<b>Date:</b> {patient_info['date']}", styles["Normal"]))
        story.append(Paragraph(f"<b>Patient Name:</b> {patient_info['name']}", styles["Normal"]))
        story.append(Paragraph(f"<b>Age:</b> {patient_info['age']} | <b>Gender:</b> {patient_info['gender']}", styles["Normal"]))
        story.append(Spacer(1, 0.3*inch))
        
        # Diagnosis
        story.append(Paragraph(f"<b>Diagnosis:</b> {prescription_data['diagnosis']}", styles["Normal"]))
        story.append(Paragraph(f"<b>Risk Level:</b> {prescription_data['risk_level']}", styles["Normal"]))
        story.append(Spacer(1, 0.3*inch))
        
        # Medications Table
        story.append(Paragraph("<b>PRESCRIBED MEDICATIONS:</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.1*inch))
        
        med_data = [["Medication", "Dosage", "Frequency", "Duration"]]
        for med in prescription_data["medications"]:
            med_data.append([
                med["name"],
                med["dosage"],
                med["frequency"],
                med["duration"]
            ])
        
        med_table = Table(med_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        med_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00c9a7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")])
        ]))
        story.append(med_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Instructions
        story.append(Paragraph("<b>INSTRUCTIONS:</b>", styles["Heading2"]))
        for instruction in prescription_data["instructions"]:
            story.append(Paragraph(f"• {instruction}", styles["Normal"]))
        story.append(Spacer(1, 0.2*inch))
        
        # Warnings
        story.append(Paragraph("<b>WARNINGS:</b>", styles["Heading2"]))
        for warning in prescription_data["warnings"]:
            story.append(Paragraph(f"{warning}", styles["Normal"]))
        story.append(Spacer(1, 0.2*inch))
        
        # Follow-up
        story.append(Paragraph(f"<b>Follow-up:</b> {prescription_data['follow_up']}", styles["Normal"]))
        story.append(Spacer(1, 0.4*inch))
        
        # Footer
        story.append(Paragraph("_" * 50, styles["Normal"]))
        story.append(Paragraph("<b>Dr. Rajiv Sharma, MD</b>", styles["Normal"]))
        story.append(Paragraph("Oncologist | AIIMS Delhi", styles["Normal"]))
        story.append(Paragraph("License No: MED-2024-12345", styles["Normal"]))
        
        # Build PDF
        doc.build(story)
        
        return send_file(pdf_path, as_attachment=True, download_name=f"prescription_{patient_info['prescription_id']}.pdf")
        
    except Exception as e:
        print(f"Error generating prescription PDF: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for("prescription"))




@app.route("/test-dietary")
def test_dietary():
    """Test route to verify dietary recommendations work"""
    pred = 1  # Test with Malignant
    risk_level = "High"
    
    dietary_data = get_dietary_recommendations(pred, risk_level)
    
    return render_template(
        "dietary.html",
        result="Malignant",
        **dietary_data
    )

# ── HISTORY ───────────────────────────────────────────────────────────────────

@app.route("/history")
def history():
    return render_template("history.html", history=history_data)


@app.route("/delete/<int:record_id>")
def delete(record_id):
    history_data[:] = [x for x in history_data if x["id"] != record_id]
    return redirect(url_for("history"))


# ── APPOINTMENT ───────────────────────────────────────────────────────────────

@app.route("/appointment")
def appointment():
    return render_template("appointment.html")



@app.route("/book", methods=["POST"])
def book():
    name = request.form.get("name", "")
    phone = request.form.get("phone", "")
    email = request.form.get("email", "")
    date = request.form.get("date", "")
    doctor = request.form.get("doctor", "")
    time = request.form.get("time", "")
    
    # Store appointment data
    appointments_data.append({
        "id": len(appointments_data) + 1,
        "name": name,
        "phone": phone,
        "email": email,
        "date": date,
        "doctor": doctor,
        "time": time,
        "booked_on": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    flash(f"✅ {name}, your appointment with {doctor} is confirmed on {date} at {time}.", "success")
    return redirect(url_for("appointments_list"))

@app.route("/appointments")
def appointments_list():
    return render_template("appointments_list.html", appointments=appointments_data)




# ── UPLOAD CSV ────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file:
        flash("No file selected.", "error")
        return redirect(url_for("dashboard"))

    try:   # ✅ NOW INSIDE FUNCTION

        df = pd.read_csv(file)

        df.columns = df.columns.str.strip().str.lower()

        col_map = {
            "radius": "radius_mean",
            "texture": "texture_mean",
            "perimeter": "perimeter_mean",
            "area": "area_mean"
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        CSV_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']

        missing = [col for col in CSV_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        inputs = df[CSV_FEATURES]

        scaled_inputs = scaler.transform(inputs)
        predictions = model.predict(scaled_inputs)

        df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in predictions]

        benign = int((predictions == 0).sum())
        malignant = int((predictions == 1).sum())

        csv_path = os.path.join(app.config["UPLOAD_FOLDER"], "prediction_result.csv")
        df.to_csv(csv_path, index=False)

        return render_template(
            "dashboard.html",
            table=df.to_html(classes="result-table", index=False),
            download_link="/uploads/prediction_result.csv",
            benign=benign,
            malignant=malignant
        )

    except Exception as e:
        flash(f"Error processing file: {str(e)}", "error")
        return redirect(url_for("dashboard"))       


# ── PDF DOWNLOAD ──────────────────────────────────────────────────────────────

@app.route("/download_pdf")
def download_pdf():
    if "user" not in session:
        return redirect(url_for("login"))
    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], "prediction_result.csv")
    if not os.path.exists(csv_path):
        flash("No data found. Please upload a CSV first.", "error")
        return redirect(url_for("dashboard"))

    df = pd.read_csv(csv_path)
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "prediction_result.pdf")
    doc = SimpleDocTemplate(pdf_path)
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00c9a7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    doc.build([table])
    return send_file(pdf_path, as_attachment=True, download_name="prediction_report.pdf")


# ── REPORT DOWNLOAD ───────────────────────────────────────────────────────────

@app.route("/download-report", methods=["POST"])
def download_report():
    data = request.json or {}
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "report.pdf")
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"Patient Name: {data.get('name', 'N/A')}", styles["Normal"]),
        Paragraph(f"Age: {data.get('age', 'N/A')}", styles["Normal"]),
        Paragraph(f"Result: {data.get('prediction', 'N/A')}", styles["Normal"]),
        Paragraph(f"Confidence: {data.get('probability', 'N/A')}", styles["Normal"]),
    ]
    doc.build(content)
    return send_file(pdf_path, as_attachment=True)

# ── CHATBOT WITH GEMINI ──────────────────────────────────────────────────────

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY not found. Gemini chatbot will be disabled.")

# Initialize Gemini model
try:
    if GEMINI_API_KEY:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_ENABLED = True
        print("✅ Gemini AI initialized successfully!")
    else:
        gemini_model = None
        GEMINI_ENABLED = False

except Exception as e:
    print(f"❌ Gemini initialization failed: {e}")
    gemini_model = None
    GEMINI_ENABLED = False

# Store chat sessions per user
chat_sessions = {}

def get_or_create_chat_session(session_id):
    """Get existing chat session or create new one"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_model.start_chat(history=[])
    return chat_sessions[session_id]


# Comprehensive medical and project-related responses
CHAT_RESPONSES = {
    # Breast Cancer Symptoms
    "symptom": "Common breast cancer symptoms include:\n• A lump or mass in the breast\n• Change in breast size or shape\n• Skin dimpling or puckering\n• Nipple discharge (especially bloody)\n• Nipple inversion\n• Redness or scaling of breast skin\n• Swelling in armpit\n\nIf you notice any of these, consult a doctor immediately.",
    
    "symptoms": "Common breast cancer symptoms include:\n• A lump or mass in the breast\n• Change in breast size or shape\n• Skin dimpling or puckering\n• Nipple discharge (especially bloody)\n• Nipple inversion\n• Redness or scaling of breast skin\n• Swelling in armpit\n\nIf you notice any of these, consult a doctor immediately.",
    
    # Benign vs Malignant
    "benign": "Benign means NON-CANCEROUS:\n• Cells are not spreading to other tissues\n• Growth is usually slow and contained\n• Not life-threatening but needs monitoring\n• Regular check-ups every 6 months recommended\n• May require biopsy to confirm\n\nStill important to follow up with your doctor!",
    
    "malignant": "Malignant means CANCEROUS:\n• Cancer cells can spread to other tissues\n• Requires immediate medical attention\n• Treatment options: surgery, chemotherapy, radiation\n• Early detection greatly improves outcomes\n• Consult an oncologist immediately\n\n⚠️ This is serious - please see a specialist right away!",
    
    # Cancer General
    "cancer": "Breast cancer is a disease where breast cells grow uncontrollably and form tumors.\n\n📊 Key Facts:\n• Most common cancer in women worldwide\n• Early detection saves lives (90%+ survival rate when caught early)\n• Risk factors: age, family history, genetics, lifestyle\n• Screening: mammograms, self-exams, clinical exams\n\n💪 Remember: Early detection is key!",
    
    "breast cancer": "Breast cancer is a disease where breast cells grow uncontrollably and form tumors.\n\n📊 Key Facts:\n• Most common cancer in women worldwide\n• Early detection saves lives (90%+ survival rate when caught early)\n• Risk factors: age, family history, genetics, lifestyle\n• Screening: mammograms, self-exams, clinical exams\n\n💪 Remember: Early detection is key!",
    
    # Treatment
    "treatment": "Breast cancer treatment options include:\n\n🏥 Surgery:\n• Lumpectomy (removes tumor only)\n• Mastectomy (removes entire breast)\n\n💊 Medication:\n• Chemotherapy (kills cancer cells)\n• Hormone therapy (blocks hormones)\n• Targeted therapy (attacks specific cancer cells)\n\n⚡ Radiation:\n• Kills remaining cancer cells after surgery\n\n👨⚕️ Treatment depends on cancer stage, type, and patient health. Always consult your oncologist!",
    
    "chemotherapy": "Chemotherapy uses drugs to kill cancer cells throughout the body.\n\n💊 Common drugs:\n• Doxorubicin (Adriamycin)\n• Cyclophosphamide (Cytoxan)\n• Paclitaxel (Taxol)\n\n⚠️ Side effects:\n• Hair loss\n• Nausea and vomiting\n• Fatigue\n• Increased infection risk\n\n✅ Most side effects are temporary and manageable with medication.",
    
    # Risk Factors
    "risk": "Breast cancer risk factors include:\n\n🧬 Genetic:\n• Family history of breast cancer\n• BRCA1/BRCA2 gene mutations\n• Age (risk increases after 50)\n\n🏃 Lifestyle:\n• Obesity\n• Alcohol consumption\n• Lack of physical activity\n• Smoking\n\n👶 Reproductive:\n• Early menstruation (before 12)\n• Late menopause (after 55)\n• No pregnancies or late first pregnancy\n\n💡 Many risk factors can be modified through lifestyle changes!",
    
    # Diagnosis
    "biopsy": "A biopsy is a procedure where tissue is removed and examined under a microscope.\n\n🔬 Types:\n• Fine needle aspiration (FNA)\n• Core needle biopsy\n• Surgical biopsy\n\n✅ Purpose:\n• Confirms if tumor is benign or malignant\n• Determines cancer type and grade\n• Guides treatment decisions\n\n⏱️ Results typically available in 3-7 days.",
    
    "mammogram": "A mammogram is an X-ray of the breast used to detect cancer early.\n\n📸 Types:\n• Screening mammogram (routine check)\n• Diagnostic mammogram (investigates symptoms)\n\n📅 Recommendations:\n• Age 40-44: Optional annual screening\n• Age 45-54: Annual screening\n• Age 55+: Every 1-2 years\n\n💡 Can detect cancer 2-3 years before you can feel a lump!",
    
    "diagnosis": "Breast cancer diagnosis involves several steps:\n\n1️⃣ Physical Examination\n2️⃣ Imaging Tests (mammogram, ultrasound, MRI)\n3️⃣ Biopsy (tissue sample analysis)\n4️⃣ Lab Tests (hormone receptors, HER2 status)\n5️⃣ Staging (determine cancer spread)\n\n📊 Staging (0-IV) determines treatment approach.\n\nEarly detection through regular screening is crucial!",
    
    # Diet & Lifestyle
    "diet": "Healthy diet recommendations for breast cancer:\n\n✅ EAT MORE:\n• Fruits and vegetables (5+ servings daily)\n• Whole grains (brown rice, quinoa, oats)\n• Lean proteins (fish, chicken, beans)\n• Healthy fats (olive oil, avocado, nuts)\n\n❌ AVOID:\n• Processed meats\n• Refined sugars\n• Alcohol\n• Fried and fast foods\n\n💧 Stay hydrated: 8-10 glasses of water daily!",
    
    "food": "Healthy diet recommendations for breast cancer:\n\n✅ EAT MORE:\n• Fruits and vegetables (5+ servings daily)\n• Whole grains (brown rice, quinoa, oats)\n• Lean proteins (fish, chicken, beans)\n• Healthy fats (olive oil, avocado, nuts)\n\n❌ AVOID:\n• Processed meats\n• Refined sugars\n• Alcohol\n• Fried and fast foods\n\n💧 Stay hydrated: 8-10 glasses of water daily!",
    
    "exercise": "Exercise benefits during and after breast cancer treatment:\n\n💪 Benefits:\n• Reduces fatigue\n• Improves mood and mental health\n• Maintains healthy weight\n• Reduces risk of recurrence\n• Strengthens immune system\n\n🏃 Recommendations:\n• 150 minutes moderate activity per week\n• Walking, swimming, yoga, cycling\n• Start slowly and gradually increase\n• Listen to your body\n\n⚠️ Consult your doctor before starting any exercise program!",
    
    # About the Project
    "project": "This is a Breast Cancer Prediction System that uses Machine Learning to analyze patient data.\n\n🎯 Features:\n• Predicts if tumor is Benign or Malignant\n• Risk level assessment (Low/Medium/High)\n• Doctor recommendations\n• Dietary guidelines\n• Prescription generation\n• Patient history tracking\n• AI-powered chatbot\n\n🤖 Technology: Python, Flask, Machine Learning, Google Gemini AI\n\n💡 Purpose: Early detection and patient education to save lives!",
    
    "app": "This is a Breast Cancer Prediction System that uses Machine Learning to analyze patient data.\n\n🎯 Features:\n• Predicts if tumor is Benign or Malignant\n• Risk level assessment (Low/Medium/High)\n• Doctor recommendations\n• Dietary guidelines\n• Prescription generation\n• Patient history tracking\n• AI-powered chatbot\n\n🤖 Technology: Python, Flask, Machine Learning, Google Gemini AI\n\n💡 Purpose: Early detection and patient education to save lives!",
    
    "how does this work": "This system works in 4 simple steps:\n\n1️⃣ INPUT: Enter patient data (radius, texture, perimeter, area)\n2️⃣ ANALYSIS: Machine Learning model analyzes the data\n3️⃣ PREDICTION: System predicts Benign or Malignant with confidence score\n4️⃣ RECOMMENDATIONS: Provides doctor recommendations, diet plans, and prescriptions\n\n🤖 The ML model was trained on real breast cancer data to achieve high accuracy!\n\n📊 You can also upload CSV files for batch predictions!",
    
    "features": "🎯 System Features:\n\n✅ Prediction System:\n• Single patient prediction\n• Bulk CSV upload\n• Confidence scoring\n• Risk level assessment\n\n📋 Reports:\n• Detailed result reports\n• Dietary recommendations\n• Medication prescriptions\n• PDF downloads\n\n👥 Management:\n• Patient history tracking\n• Appointment booking\n• Doctor recommendations\n\n💬 AI Chatbot:\n• 24/7 medical information\n• Instant answers\n• Patient support",
    
    # Greetings
    "hello": "Hello! 👋 I'm your Breast Cancer Assistant.\n\nI can help you with:\n• Breast cancer information\n• Symptoms and diagnosis\n• Treatment options\n• Diet and lifestyle advice\n• Understanding your results\n• Information about this system\n\nWhat would you like to know?",
    
    "hi": "Hi there! 👋 I'm here to help you with breast cancer information and support.\n\nAsk me about:\n• Symptoms and warning signs\n• Treatment options\n• Diet recommendations\n• Risk factors\n• How this prediction system works\n\nWhat can I help you with today?",
    
    "help": "I can answer questions about:\n\n🏥 Medical Topics:\n• Breast cancer symptoms\n• Benign vs Malignant\n• Treatment options\n• Risk factors\n• Diagnosis procedures\n• Diet and exercise\n\n💻 This System:\n• How predictions work\n• System features\n• How to use the app\n\n💬 Just ask your question in plain English!",
    
    # Prevention
    "prevention": "Breast cancer prevention tips:\n\n✅ DO:\n• Maintain healthy weight (BMI 18.5-24.9)\n• Exercise regularly (150 min/week)\n• Eat healthy diet (fruits, vegetables, whole grains)\n• Limit alcohol (max 1 drink/day)\n• Breastfeed if possible\n• Regular self-exams\n• Annual mammograms after 40\n\n❌ AVOID:\n• Smoking\n• Excessive alcohol\n• Obesity\n• Sedentary lifestyle\n\n🔍 Early detection is the best protection!",
    
    "screening": "Breast cancer screening guidelines:\n\n📅 Age 20-39:\n• Monthly self-exams\n• Clinical exam every 3 years\n\n📅 Age 40-44:\n• Optional annual mammogram\n• Monthly self-exams\n\n📅 Age 45-54:\n• Annual mammogram\n• Monthly self-exams\n\n📅 Age 55+:\n• Mammogram every 1-2 years\n• Continue self-exams\n\n⚠️ High-risk individuals may need earlier/more frequent screening!",
    
    # Support
    "scared": "It's completely normal to feel scared. Remember:\n\n💪 You're not alone:\n• Millions of survivors worldwide\n• Support groups available\n• Family and friends are there for you\n\n✅ Positive facts:\n• 90%+ survival rate when caught early\n• Treatment has improved dramatically\n• Many people live full, normal lives after treatment\n\n🤗 Take it one day at a time. Focus on what you can control: following treatment, staying positive, and taking care of yourself.\n\nConsult your doctor for personalized support and counseling resources.",
    
    "support": "Support resources for breast cancer patients:\n\n🤝 Support Groups:\n• Local cancer support groups\n• Online communities\n• Survivor networks\n\n📞 Helplines:\n• Cancer helplines (24/7)\n• Counseling services\n• Financial assistance programs\n\n👨👩👧👦 Family Support:\n• Family counseling\n• Caregiver resources\n• Children's support programs\n\n💻 Online Resources:\n• Cancer.org\n• BreastCancer.org\n• National Cancer Institute\n\nYou don't have to face this alone!",
}


def is_medical_question(user_message):
    """Check if question is medical/breast cancer related"""
    medical_keywords = [
        "cancer", "breast", "tumor", "tumour", "symptom", "sign", "benign", "malignant",
        "treatment", "cure", "therapy", "chemo", "chemotherapy", "surgery", "radiation",
        "biopsy", "mammogram", "diagnosis", "risk", "prevent", "screening", "lump",
        "pain", "discharge", "doctor", "oncologist", "medical", "health", "disease",
        "diet", "food", "exercise", "scared", "worried", "support", "help"
    ]
    
    msg_lower = user_message.lower()
    return any(keyword in msg_lower for keyword in medical_keywords)


def find_best_match(user_message):
    """Find the best matching response using flexible keyword matching with similarity"""
    
    msg_lower = user_message.lower().strip()
    
    # STEP 1: Check for EXACT phrase matches first (highest priority)
    exact_matches = {
        "how does this work": "how does this work",
        "how does it work": "how does this work",
        "how work": "how does this work",
        "how it work": "how does this work",
        "how this work": "how does this work",
        "explain how it works": "how does this work",
        "what to eat": "diet",
        "what should i eat": "diet",
        "what should eat": "diet",
        "what can i eat": "diet",
        "what food": "food",
        "food to eat": "diet",
        "tell me about food": "diet",
        "nutrition advice": "diet",
        "what are symptoms": "symptoms",
        "what are the symptoms": "symptoms",
        "tell me symptoms": "symptoms",
        "show symptoms": "symptoms",
        "show me symptoms": "symptoms",
        "signs of cancer": "symptoms",
        "cancer signs": "symptoms",
        "is benign dangerous": "benign",
        "what is benign": "benign",
        "benign dangerous": "benign",
        "tell me about benign": "benign",
        "explain benign": "benign",
        "what is malignant": "malignant",
        "tell me about malignant": "malignant",
        "explain malignant": "malignant",
        "what is cancer": "cancer",
        "tell me about cancer": "cancer",
        "explain cancer": "cancer",
        "what is breast cancer": "breast cancer",
        "tell me about breast cancer": "breast cancer",
        "how to prevent": "prevention",
        "how can i prevent": "prevention",
        "how prevent cancer": "prevention",
        "prevention tips": "prevention",
        "what are features": "features",
        "system features": "features",
        "app features": "features",
        "what can this do": "features",
        "hello": "hello",
        "hi": "hi",
        "hey": "hello",
        "hye": "hi",
        "help": "help",
        "help me": "help",
        "good morning": "hello",
        "good afternoon": "hello",
        "good evening": "hello",
        "what is treatment": "treatment",
        "treatment options": "treatment",
        "how to treat": "treatment",
        "what is chemo": "chemotherapy",
        "chemotherapy info": "chemotherapy",
        "what is biopsy": "biopsy",
        "biopsy info": "biopsy",
        "what is mammogram": "mammogram",
        "mammogram info": "mammogram",
        "risk factors": "risk",
        "what causes cancer": "risk",
        "exercise tips": "exercise",
        "workout advice": "exercise",
        "i am scared": "scared",
        "i am worried": "scared",
        "i am afraid": "scared",
        "need support": "support",
        "where to get help": "support",
    }
    
    # Check exact matches
    for phrase, response_key in exact_matches.items():
        if phrase == msg_lower or phrase in msg_lower:
            print(f"✅ Exact match found: '{phrase}' -> '{response_key}'")
            return response_key
    
    # STEP 2: Check for direct key matches in CHAT_RESPONSES
    for key in CHAT_RESPONSES.keys():
        if key == msg_lower or key in msg_lower:
            print(f"✅ Direct key match: '{key}'")
            return key
    
    # STEP 3: Comprehensive keyword group matching with FLEXIBLE scoring
    keyword_groups = {
        "symptoms": [
            "symptom", "sign", "warning", "notice", "detect", "feel", "lump", "pain", 
            "discharge", "change", "indication", "show", "display", "manifest"
        ],
        "benign": [
            "benign", "non-cancer", "non cancer", "not cancer", "harmless", 
            "non-malignant", "safe", "non dangerous"
        ],
        "malignant": [
            "malignant", "cancerous", "tumor", "tumour", "dangerous", "serious", 
            "harmful", "life threatening"
        ],
        "cancer": [
            "what is cancer", "about cancer", "cancer info", "tell me about cancer", 
            "explain cancer", "cancer disease", "cancer condition"
        ],
        "breast cancer": [
            "breast cancer", "breast tumor", "breast disease", "mammary cancer",
            "breast malignancy"
        ],
        "treatment": [
            "treatment", "treat", "cure", "therapy", "medicine", "medication", 
            "heal", "surgery", "operation", "remedy", "medical care"
        ],
        "chemotherapy": [
            "chemo", "chemotherapy", "chemical", "drug treatment", "cytotoxic",
            "cancer drug", "chemo treatment"
        ],
        "risk": [
            "risk", "cause", "factor", "chance", "probability", "likely", 
            "prone", "susceptible", "reason", "why cancer"
        ],
        "biopsy": [
            "biopsy", "test", "sample", "tissue", "needle", "examine",
            "tissue test", "lab test"
        ],
        "mammogram": [
            "mammogram", "x-ray", "xray", "scan", "imaging", "screening test",
            "breast scan", "breast xray"
        ],
        "diagnosis": [
            "diagnosis", "diagnose", "find out", "identify", "confirm",
            "detect cancer", "how to know"
        ],
        "diet": [
            "diet", "food", "eat", "nutrition", "meal", "consume", "avoid eating", 
            "should i eat", "what to eat", "eating", "foods", "dietary", "nutritional"
        ],
        "exercise": [
            "exercise", "workout", "physical activity", "gym", "fitness", "active", 
            "movement", "training", "sport", "yoga"
        ],
        "project": [
            "project", "app", "application", "system", "website", "platform", 
            "tool", "software", "program"
        ],
        "how does this work": [
            "how work", "how does", "how it work", "explain", "understand", 
            "what does this do", "functionality", "how this", "working"
        ],
        "features": [
            "feature", "capability", "function", "what can", "able to do", 
            "options", "abilities", "functions"
        ],
        "hello": [
            "hello", "greetings", "good morning", "good afternoon", "good evening"
        ],
        "hi": [
            "hi", "hey", "hye", "hii", "heya"
        ],
        "help": [
            "help", "assist", "support", "guide", "info", "information", 
            "tell me about", "need info"
        ],
        "prevention": [
            "prevent", "avoid", "stop", "reduce risk", "lower chance", "protect",
            "prevention", "preventive", "how to avoid"
        ],
        "screening": [
            "screening", "check", "checkup", "examination", "regular test", 
            "when should", "testing", "check up"
        ],
        "scared": [
            "scared", "afraid", "fear", "worried", "anxious", "nervous", 
            "terrified", "panic", "frightened"
        ],
        "support": [
            "support", "help me", "need help", "resources", "where to go", 
            "who can help", "assistance", "counseling"
        ],
    }
    
    # Score each keyword group with FLEXIBLE matching
    scores = {}
    for key, keywords in keyword_groups.items():
        score = 0
        matches = []
        
        for keyword in keywords:
            # Check if keyword is in message
            if keyword in msg_lower:
                # Give higher score for longer/more specific matches
                keyword_score = len(keyword.split()) * 10
                score += keyword_score
                matches.append(keyword)
        
        if score > 0:
            scores[key] = score
            print(f"  📊 '{key}' scored {score} (matched: {matches})")
    
    # Return the keyword with highest score
    if scores:
        best_match = max(scores, key=scores.get)
        print(f"✅ Best keyword match: '{best_match}' (score: {scores[best_match]})")
        return best_match
    
    print("❌ No match found in stored responses")
    return None


def get_chat_response(user_message, session_id="anonymous"):
    """HYBRID: Get response from stored data OR Gemini AI for ANY question"""
    
    if not user_message:
        return "Please ask me a question! I can help with breast cancer information, treatment options, diet advice, and more."
    
    print(f"\n🔍 Processing message: '{user_message}'")
    
    # STEP 1: Try to find best match in stored responses FIRST
    best_match = find_best_match(user_message)
    
    if best_match and best_match in CHAT_RESPONSES:
        print(f"✅ Found stored response for: '{best_match}'")
        return CHAT_RESPONSES[best_match]
    
    # STEP 2: If no stored match found, use Gemini AI for ANY question
    print("🤖 No stored match - trying Gemini AI for general response")
    gemini_response = try_gemini_response(user_message, session_id)
    
    if gemini_response:
        print("✅ Gemini AI response received")
        return gemini_response
    
    # STEP 3: Only if BOTH fail, return helpful fallback
    print("⚠️ Both stored responses and Gemini failed - using fallback")
    return """I'm here to help with breast cancer information! 

I can answer questions about:
• Symptoms and warning signs
• Benign vs Malignant tumors
• Treatment options (surgery, chemotherapy, radiation)
• Risk factors and prevention
• Diet and exercise recommendations
• Diagnosis procedures (biopsy, mammogram)
• How this prediction system works

Try asking:
• "What are the symptoms?"
• "Is benign dangerous?"
• "How can I prevent cancer?"
• "What should I eat?"
• "How does this work?"

What would you like to know?"""



def try_gemini_response(user_message, session_id):
    """Try to get response from Gemini AI"""
    
    if not GEMINI_ENABLED:
        return None
    
    try:
        # Get or create chat session
        chat = get_or_create_chat_session(session_id)
        
        # Add medical context for better responses
        prompt = f"""You are a helpful medical AI assistant specializing in breast cancer information.

User question: {user_message}

Provide a clear, concise, and accurate response (2-4 sentences). 
For medical advice, always recommend consulting healthcare professionals.
Be empathetic and supportive."""
        
        # Send to Gemini
        response = chat.send_message(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        
        return None
        
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return None



@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages with HYBRID approach"""
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    
    print(f"\n💬 Chat request: {user_message}")
    
    if not user_message:
        return jsonify({"response": "Please ask me a question!"})
    
    # Get session ID
    session_id = session.get("user", "anonymous")
    
    # Get response using hybrid approach
    reply = get_chat_response(user_message, session_id)
    
    print(f"✅ Sending response: {reply[:100]}...")
    
    return jsonify({"response": reply})


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    """Clear chat history"""
    session_id = session.get("user", "anonymous")
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return jsonify({"status": "cleared", "message": "Chat cleared!"})


@app.route("/test-gemini")
def test_gemini():
    """Test Gemini API"""
    try:
        test_response = gemini_model.generate_content("Say hello in one sentence")
        return jsonify({
            "status": "success",
            "enabled": GEMINI_ENABLED,
            "response": test_response.text if test_response else "No response",
            "message": "✅ Gemini is working!"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "enabled": GEMINI_ENABLED,
            "error": str(e),
            "message": "❌ Gemini failed - using stored responses"
        })





# ── JSON API ──────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    raw_features = data.get("features", [])
    if len(raw_features) != 4:
        return jsonify({"error": "Provide exactly 4 features: radius, texture, perimeter, area"}), 400
    try:
        features = [float(v) for v in raw_features]
        if any(math.isnan(v) or math.isinf(v) or v < 0 for v in features):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "All features must be valid positive numbers"}), 400

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    probability = float(model.predict_proba(scaled)[0][1])
    risk_score, risk_level, factors = clinical_risk_analysis(features)
    return jsonify({
        "prediction": "Malignant" if prediction == 1 else "Benign",
        "confidence": round(probability * 100, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "factors": factors
    })
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

