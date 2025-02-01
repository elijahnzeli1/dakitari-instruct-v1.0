import json
import tensorflow as tf
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class MedicalConversationProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_medical_qa(self):
        """Load medical QA dataset"""
        print("Loading medical QA data...")
        data = self._create_medical_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Clean and format
        df = self._clean_data(df)
        
        print(f"Total examples after processing: {len(df)}")
        return df
    
    def _create_medical_data(self):
        """Create a comprehensive medical QA dataset"""
        return [
            {
        "question": "What is the purpose of this healthcare model?",
        "answer": "This healthcare model is designed to provide accurate, reliable, and evidence-based medical information to assist users in understanding symptoms, managing conditions, and making informed health decisions. It is not a substitute for professional medical advice but serves as a helpful resource.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of diabetes?",
        "answer": "Symptoms of diabetes include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 and Type 2 diabetes may present differently.",
        "source": "medical"
    },
    {
        "question": "How can I prevent the flu?",
        "answer": "To prevent the flu: 1) Get an annual flu vaccine, 2) Wash hands frequently, 3) Avoid close contact with sick individuals, 4) Cover your mouth when coughing or sneezing, 5) Maintain a healthy lifestyle with proper diet and exercise.",
        "source": "medical"
    },
    {
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise improves cardiovascular health, strengthens muscles and bones, boosts mental health, aids in weight management, enhances sleep quality, and reduces the risk of chronic diseases like diabetes and hypertension.",
        "source": "medical"
    },
    {
        "question": "What is a healthy diet?",
        "answer": "A healthy diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. It limits processed foods, added sugars, and excessive sodium. Portion control and hydration are also important.",
        "source": "medical"
    },
    {
        "question": "What are the risks of smoking?",
        "answer": "Smoking increases the risk of lung cancer, heart disease, stroke, respiratory diseases, and other cancers. It also harms nearly every organ in the body and reduces overall life expectancy.",
        "source": "medical"
    },
    {
        "question": "How can I reduce stress?",
        "answer": "Reduce stress by practicing mindfulness, exercising regularly, getting adequate sleep, maintaining a healthy diet, engaging in hobbies, seeking social support, and using relaxation techniques like deep breathing or meditation.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of depression?",
        "answer": "Symptoms of depression include persistent sadness, loss of interest in activities, fatigue, changes in appetite or weight, difficulty sleeping, feelings of worthlessness, difficulty concentrating, and thoughts of death or suicide.",
        "source": "medical"
    },
    {
        "question": "How can I lower my cholesterol?",
        "answer": "Lower cholesterol by eating a heart-healthy diet (low in saturated fats and cholesterol), exercising regularly, maintaining a healthy weight, quitting smoking, and taking prescribed medications if necessary.",
        "source": "medical"
    },
    {
        "question": "What are the signs of a stroke?",
        "answer": "Signs of a stroke include sudden numbness or weakness in the face, arm, or leg (especially on one side), confusion, trouble speaking, vision problems, dizziness, and severe headache. Remember 'FAST': Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.",
        "source": "medical"
    },
    {
        "question": "How can I treat a sunburn?",
        "answer": "Treat a sunburn by: 1) Applying cool compresses, 2) Using aloe vera or moisturizing lotions, 3) Taking over-the-counter pain relievers, 4) Staying hydrated, 5) Avoiding further sun exposure. Seek medical help for severe burns or blisters.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of anxiety?",
        "answer": "Symptoms of anxiety include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances, and panic attacks. Severe cases may require professional treatment.",
        "source": "medical"
    },
    {
        "question": "How can I improve my digestion?",
        "answer": "Improve digestion by eating fiber-rich foods, staying hydrated, exercising regularly, avoiding processed foods, eating slowly, managing stress, and incorporating probiotics into your diet.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of asthma?",
        "answer": "Symptoms of asthma include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early morning. Triggers can include allergens, exercise, and cold air.",
        "source": "medical"
    },
    {
        "question": "How can I prevent osteoporosis?",
        "answer": "Prevent osteoporosis by consuming adequate calcium and vitamin D, engaging in weight-bearing exercises, avoiding smoking, limiting alcohol, and maintaining a healthy weight.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a urinary tract infection (UTI)?",
        "answer": "Symptoms of a UTI include a strong urge to urinate, burning sensation during urination, cloudy or bloody urine, pelvic pain, and frequent urination in small amounts. Seek medical treatment if symptoms persist.",
        "source": "medical"
    },
    {
        "question": "How can I manage allergies?",
        "answer": "Manage allergies by avoiding allergens, using antihistamines or nasal sprays, keeping windows closed during high pollen seasons, using air purifiers, and consulting a doctor for severe cases.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a concussion?",
        "answer": "Symptoms of a concussion include headache, confusion, dizziness, nausea, memory problems, sensitivity to light or noise, and balance issues. Seek immediate medical attention if a concussion is suspected.",
        "source": "medical"
    },
    {
        "question": "How can I maintain healthy skin?",
        "answer": "Maintain healthy skin by staying hydrated, eating a balanced diet, protecting against sun exposure, cleansing gently, moisturizing regularly, and avoiding smoking and excessive alcohol.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of anemia?",
        "answer": "Symptoms of anemia include fatigue, weakness, pale skin, shortness of breath, dizziness, cold hands and feet, and headaches. Treatment depends on the underlying cause.",
        "source": "medical"
    },
    {
        "question": "How can I prevent food poisoning?",
        "answer": "Prevent food poisoning by practicing proper food hygiene: washing hands, cooking food thoroughly, avoiding cross-contamination, refrigerating perishables promptly, and avoiding expired or suspicious foods.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of hypothyroidism?",
        "answer": "Symptoms of hypothyroidism include fatigue, weight gain, cold intolerance, dry skin, hair loss, constipation, depression, and slowed heart rate. Treatment involves thyroid hormone replacement.",
        "source": "medical"
    },
    {
        "question": "How can I manage chronic pain?",
        "answer": "Manage chronic pain through a combination of medications, physical therapy, exercise, stress management, acupuncture, and cognitive-behavioral therapy. Consult a healthcare provider for a personalized plan.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a panic attack?",
        "answer": "Symptoms of a panic attack include rapid heartbeat, sweating, trembling, shortness of breath, chest pain, nausea, dizziness, and fear of losing control or dying. Seek professional help if frequent.",
        "source": "medical"
    },
    {
        "question": "How can I improve my mental health?",
        "answer": "Improve mental health by staying connected with loved ones, exercising regularly, eating a balanced diet, getting enough sleep, practicing mindfulness, and seeking professional help when needed.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a migraine?",
        "answer": "Symptoms of a migraine include severe throbbing headache, nausea, vomiting, sensitivity to light and sound, and visual disturbances (aura). Treatment may involve medications and lifestyle changes.",
        "source": "medical"
    },
    {
        "question": "How can I prevent cavities?",
        "answer": "Prevent cavities by brushing twice daily with fluoride toothpaste, flossing daily, limiting sugary foods and drinks, drinking fluoridated water, and visiting the dentist regularly.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a sinus infection?",
        "answer": "Symptoms of a sinus infection include facial pain, nasal congestion, thick nasal discharge, reduced sense of smell, cough, and fatigue. Treatment may involve decongestants, saline rinses, or antibiotics.",
        "source": "medical"
    },
    {
        "question": "How can I manage arthritis pain?",
        "answer": "Manage arthritis pain by staying active, maintaining a healthy weight, using heat or cold therapy, taking prescribed medications, and practicing joint-friendly exercises like swimming or yoga.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a vitamin D deficiency?",
        "answer": "Symptoms of vitamin D deficiency include fatigue, bone pain, muscle weakness, mood changes, and frequent infections. Treatment involves sunlight exposure, dietary changes, and supplements if needed.",
        "source": "medical"
    },
    {
        "question": "What are the common symptoms of COVID-19?",
        "answer": "Common symptoms of COVID-19 include fever, dry cough, fatigue, loss of taste or smell, body aches, and difficulty breathing. Some people may also experience headaches, sore throat, and gastrointestinal issues.",
        "source": "medical"
    },
    {
        "question": "How can I manage high blood pressure?",
        "answer": "High blood pressure can be managed through lifestyle changes such as regular exercise, maintaining a healthy diet low in sodium, limiting alcohol, quitting smoking, and stress management. Medications may also be prescribed by your doctor if needed.",
        "source": "medical"
    },
    {
        "question": "What should I do for a fever?",
        "answer": "For a fever, you can: 1) Take over-the-counter medications like acetaminophen or ibuprofen, 2) Stay hydrated by drinking plenty of fluids, 3) Rest, 4) Use a cool compress, 5) Dress in light clothing. Seek medical attention if fever is very high or persists.",
        "source": "medical"
    },
    {
        "question": "What are the signs of a heart attack?",
        "answer": "Common signs of a heart attack include chest pain or pressure, pain radiating to arm/jaw/back, shortness of breath, cold sweats, nausea, and lightheadedness. Women may experience different symptoms like fatigue and indigestion. Seek immediate medical attention if suspected.",
        "source": "medical"
    },
    {
        "question": "How can I manage type 2 diabetes?",
        "answer": "Type 2 diabetes management includes: 1) Regular blood sugar monitoring, 2) Healthy diet with controlled carbohydrates, 3) Regular exercise, 4) Taking prescribed medications, 5) Weight management, 6) Regular medical check-ups. Lifestyle changes are crucial for blood sugar control.",
        "source": "medical"
    },
    {
        "question": "What are common causes of headaches?",
        "answer": "Common causes of headaches include stress, dehydration, lack of sleep, eye strain, sinus problems, tension, migraines, and certain foods/drinks. Treatment depends on the type and cause but may include rest, hydration, pain relievers, and stress management.",
        "source": "medical"
    },
    {
        "question": "How can I improve my sleep?",
        "answer": "To improve sleep: 1) Maintain a regular sleep schedule, 2) Create a relaxing bedtime routine, 3) Keep bedroom cool, dark, and quiet, 4) Avoid screens before bed, 5) Limit caffeine and alcohol, 6) Exercise regularly but not close to bedtime, 7) Manage stress through relaxation techniques.",
        "source": "medical"
    },
    {
        "question": "What are signs of dehydration?",
        "answer": "Signs of dehydration include dark urine, thirst, dry mouth, fatigue, dizziness, decreased urination, headache, and dry skin. Severe cases may show confusion, rapid heartbeat, and sunken eyes. Treatment involves fluid replacement and electrolyte balance restoration.",
        "source": "medical"
    },
    {
        "question": "How can I boost my immune system?",
        "answer": "Boost your immune system by: 1) Eating a balanced diet rich in fruits and vegetables, 2) Getting adequate sleep, 3) Regular exercise, 4) Managing stress, 5) Staying hydrated, 6) Maintaining good hygiene, 7) Taking vitamins if deficient, 8) Limiting alcohol and avoiding smoking.",
        "source": "medical"
    },
    {
        "question": "What should I do for a sprained ankle?",
        "answer": "For a sprained ankle, follow RICE: Rest the ankle, Ice it for 15-20 minutes every 2-3 hours, Compress with an elastic bandage, and Elevate above heart level. Take OTC pain relievers if needed. Seek medical attention if severe pain or inability to bear weight.",
        "source": "medical"
    },
    {
        "question": "What are the early signs of diabetes?",
        "answer": "Early signs of diabetes include frequent urination, excessive thirst, unexplained weight loss, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections. If you notice these symptoms, consult a healthcare provider for testing.",
        "source": "medical"
    },
    {
        "question": "How can I prevent the flu?",
        "answer": "To prevent the flu, get an annual flu vaccine, wash your hands frequently, avoid close contact with sick individuals, cover your mouth when coughing or sneezing, and maintain a healthy lifestyle with proper nutrition and sleep.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a stroke?",
        "answer": "Symptoms of a stroke include sudden numbness or weakness in the face, arm, or leg (especially on one side), confusion, trouble speaking or understanding speech, vision problems, dizziness, loss of balance, and severe headache. Call emergency services immediately if these symptoms occur.",
        "source": "medical"
    },
    {
        "question": "How can I reduce cholesterol levels naturally?",
        "answer": "To reduce cholesterol naturally, eat a diet rich in fruits, vegetables, whole grains, and lean proteins, avoid trans fats, exercise regularly, maintain a healthy weight, quit smoking, and limit alcohol intake. Foods like oats, nuts, and fatty fish can also help.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of anxiety?",
        "answer": "Symptoms of anxiety include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances, and panic attacks. If symptoms persist, seek help from a mental health professional.",
        "source": "medical"
    },
    {
        "question": "How can I treat a sunburn?",
        "answer": "To treat a sunburn, apply cool compresses or take a cool bath, use aloe vera or moisturizing lotions, stay hydrated, take over-the-counter pain relievers like ibuprofen, and avoid further sun exposure. Seek medical attention for severe burns or blisters.",
        "source": "medical"
    },
    {
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise improves cardiovascular health, strengthens muscles and bones, helps with weight management, boosts mental health, reduces the risk of chronic diseases, improves sleep, and increases energy levels. Aim for at least 150 minutes of moderate exercise per week.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of depression?",
        "answer": "Symptoms of depression include persistent sadness, loss of interest in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of worthlessness, difficulty concentrating, and thoughts of death or suicide. Seek professional help if you experience these symptoms.",
        "source": "medical"
    },
    {
        "question": "How can I prevent food poisoning?",
        "answer": "To prevent food poisoning, practice good hygiene, cook foods to safe temperatures, avoid cross-contamination, refrigerate perishable foods promptly, wash fruits and vegetables thoroughly, and avoid consuming expired or improperly stored foods.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of an allergic reaction?",
        "answer": "Symptoms of an allergic reaction include hives, itching, swelling, runny nose, sneezing, watery eyes, coughing, wheezing, and in severe cases, anaphylaxis (difficulty breathing, rapid pulse, and loss of consciousness). Seek emergency care for severe reactions.",
        "source": "medical"
    },
    {
        "question": "How can I manage stress effectively?",
        "answer": "To manage stress, practice relaxation techniques like deep breathing or meditation, exercise regularly, maintain a healthy diet, get enough sleep, set realistic goals, prioritize tasks, and seek support from friends, family, or a therapist.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a urinary tract infection (UTI)?",
        "answer": "Symptoms of a UTI include a strong, persistent urge to urinate, burning sensation during urination, cloudy or bloody urine, strong-smelling urine, pelvic pain, and frequent urination in small amounts. Consult a doctor for diagnosis and treatment.",
        "source": "medical"
    },
    {
        "question": "How can I prevent osteoporosis?",
        "answer": "To prevent osteoporosis, ensure adequate calcium and vitamin D intake, engage in weight-bearing exercises, avoid smoking, limit alcohol consumption, and maintain a healthy weight. Bone density tests may be recommended for older adults.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of asthma?",
        "answer": "Symptoms of asthma include wheezing, shortness of breath, chest tightness, coughing (especially at night or early morning), and difficulty breathing. Triggers can include allergens, exercise, and cold air. Use prescribed inhalers and seek medical advice for management.",
        "source": "medical"
    },
    {
        "question": "How can I treat a minor burn at home?",
        "answer": "For a minor burn, run cool water over the area for 10-15 minutes, apply a soothing ointment like aloe vera, cover with a sterile bandage, and take over-the-counter pain relievers if needed. Avoid popping blisters and seek medical attention for severe burns.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a sinus infection?",
        "answer": "Symptoms of a sinus infection include facial pain or pressure, nasal congestion, thick nasal discharge, loss of smell, cough, fatigue, and fever. Treatment may involve decongestants, saline nasal sprays, or antibiotics if bacterial.",
        "source": "medical"
    },
    {
        "question": "How can I prevent cavities?",
        "answer": "To prevent cavities, brush your teeth twice a day with fluoride toothpaste, floss daily, limit sugary foods and drinks, use mouthwash, and visit your dentist regularly for check-ups and cleanings.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a migraine?",
        "answer": "Symptoms of a migraine include severe throbbing or pulsing pain, usually on one side of the head, nausea, vomiting, sensitivity to light and sound, and visual disturbances (aura). Treatment may involve pain relievers, triptans, and lifestyle changes.",
        "source": "medical"
    },
    {
        "question": "How can I manage acid reflux?",
        "answer": "To manage acid reflux, avoid trigger foods like spicy or fatty meals, eat smaller portions, avoid lying down after eating, elevate the head of your bed, lose weight if needed, and take antacids or prescribed medications.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of anemia?",
        "answer": "Symptoms of anemia include fatigue, weakness, pale skin, shortness of breath, dizziness, cold hands and feet, and headaches. Treatment depends on the cause and may involve iron supplements, dietary changes, or other medications.",
        "source": "medical"
    },
    {
        "question": "How can I prevent back pain?",
        "answer": "To prevent back pain, maintain good posture, exercise regularly to strengthen core muscles, lift objects properly, avoid prolonged sitting, use ergonomic furniture, and maintain a healthy weight.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a concussion?",
        "answer": "Symptoms of a concussion include headache, confusion, dizziness, nausea, memory problems, sensitivity to light or noise, and balance issues. Seek medical attention if a concussion is suspected, especially after a head injury.",
        "source": "medical"
    },
    {
        "question": "How can I manage seasonal allergies?",
        "answer": "To manage seasonal allergies, avoid allergens like pollen, use antihistamines or nasal sprays, keep windows closed during high pollen seasons, shower after being outdoors, and consider allergy shots if symptoms are severe.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a thyroid disorder?",
        "answer": "Symptoms of a thyroid disorder include fatigue, weight changes, mood swings, hair loss, temperature sensitivity, and changes in heart rate. Hypothyroidism and hyperthyroidism have different symptoms and require medical diagnosis and treatment.",
        "source": "medical"
    },
    {
        "question": "How can I prevent kidney stones?",
        "answer": "To prevent kidney stones, drink plenty of water, limit salt and animal protein, avoid foods high in oxalates (like spinach and nuts), and maintain a healthy diet rich in fruits and vegetables. Consult a doctor for personalized advice.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a panic attack?",
        "answer": "Symptoms of a panic attack include sudden intense fear, rapid heartbeat, sweating, trembling, shortness of breath, chest pain, nausea, and dizziness. Seek help from a mental health professional if panic attacks are frequent.",
        "source": "medical"
    },
    {
        "question": "How can I manage eczema?",
        "answer": "To manage eczema, moisturize regularly, avoid triggers like harsh soaps or allergens, use prescribed creams or ointments, take lukewarm baths, and wear soft, breathable fabrics. Consult a dermatologist for severe cases.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a vitamin D deficiency?",
        "answer": "Symptoms of a vitamin D deficiency include fatigue, bone pain, muscle weakness, mood changes, and frequent infections. Treatment involves increasing sun exposure, eating vitamin D-rich foods, and taking supplements if needed.",
        "source": "medical"
    },
    {
        "question": "How can I prevent eye strain?",
        "answer": "To prevent eye strain, follow the 20-20-20 rule (every 20 minutes, look at something 20 feet away for 20 seconds), adjust screen brightness, use proper lighting, and take regular breaks from screens.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a yeast infection?",
        "answer": "Symptoms of a yeast infection include itching, burning, redness, and swelling in the vaginal area, along with a thick, white discharge. Over-the-counter antifungal treatments are usually effective, but consult a doctor if symptoms persist.",
        "source": "medical"
    },
    {
        "question": "How can I manage chronic pain?",
        "answer": "To manage chronic pain, consider physical therapy, medications, relaxation techniques, exercise, acupuncture, and cognitive-behavioral therapy. Work with a healthcare provider to develop a personalized pain management plan.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a hernia?",
        "answer": "Symptoms of a hernia include a noticeable bulge, pain or discomfort (especially when lifting or bending), weakness, and pressure in the affected area. Surgery is often required to repair a hernia.",
        "source": "medical"
    },
    {
        "question": "How can I prevent varicose veins?",
        "answer": "To prevent varicose veins, exercise regularly, maintain a healthy weight, avoid prolonged standing or sitting, elevate your legs, and wear compression stockings if recommended by a doctor.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a gluten intolerance?",
        "answer": "Symptoms of gluten intolerance include bloating, diarrhea, constipation, abdominal pain, fatigue, and headaches. A gluten-free diet is the primary treatment, but consult a doctor for proper diagnosis.",
        "source": "medical"
    },
    {
        "question": "How can I manage insomnia?",
        "answer": "To manage insomnia, establish a regular sleep schedule, create a relaxing bedtime routine, avoid caffeine and screens before bed, and consider cognitive-behavioral therapy or medications if needed.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a bladder infection?",
        "answer": "Symptoms of a bladder infection include a strong urge to urinate, burning sensation during urination, cloudy or bloody urine, and pelvic discomfort. Antibiotics are typically required for treatment.",
        "source": "medical"
    },
    {
        "question": "How can I prevent gum disease?",
        "answer": "To prevent gum disease, brush and floss daily, use an antiseptic mouthwash, avoid smoking, eat a balanced diet, and visit your dentist regularly for cleanings and check-ups.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a vitamin B12 deficiency?",
        "answer": "Symptoms of a vitamin B12 deficiency include fatigue, weakness, numbness or tingling in hands and feet, difficulty walking, memory problems, and mood changes. Treatment involves dietary changes or supplements.",
        "source": "medical"
    },
    {
        "question": "How can I manage arthritis pain?",
        "answer": "To manage arthritis pain, use anti-inflammatory medications, apply heat or cold packs, exercise regularly, maintain a healthy weight, and consider physical therapy or joint surgery if necessary.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a gallbladder attack?",
        "answer": "Symptoms of a gallbladder attack include severe abdominal pain (especially after eating fatty foods), nausea, vomiting, and bloating. Seek medical attention if symptoms persist or worsen.",
        "source": "medical"
    },
    {
        "question": "How can I prevent motion sickness?",
        "answer": "To prevent motion sickness, sit in the front seat of a car or over the wing of a plane, focus on the horizon, avoid reading, and take over-the-counter medications like dimenhydrinate before traveling.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a blood clot?",
        "answer": "Symptoms of a blood clot include swelling, pain, redness, and warmth in the affected area. In the lungs, it may cause shortness of breath and chest pain. Seek immediate medical attention if a blood clot is suspected.",
        "source": "medical"
    },
    {
        "question": "How can I manage lactose intolerance?",
        "answer": "To manage lactose intolerance, avoid dairy products or use lactase supplements, choose lactose-free alternatives, and ensure adequate calcium intake through other sources like leafy greens or fortified foods.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a pinched nerve?",
        "answer": "Symptoms of a pinched nerve include pain, numbness, tingling, and weakness in the affected area. Rest, physical therapy, and anti-inflammatory medications can help, but consult a doctor for severe cases.",
        "source": "medical"
    },
    {
        "question": "How can I prevent shin splints?",
        "answer": "To prevent shin splints, wear proper footwear, stretch before exercising, avoid overtraining, and gradually increase the intensity of your workouts. Rest and ice can help if symptoms occur.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a stomach ulcer?",
        "answer": "Symptoms of a stomach ulcer include burning stomach pain, bloating, nausea, and loss of appetite. Treatment may involve medications to reduce stomach acid and antibiotics if caused by H. pylori bacteria.",
        "source": "medical"
    },
    {
        "question": "How can I manage carpal tunnel syndrome?",
        "answer": "To manage carpal tunnel syndrome, wear a wrist splint, avoid repetitive motions, take breaks during activities, and consider physical therapy or surgery if symptoms are severe.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn meniscus?",
        "answer": "Symptoms of a torn meniscus include knee pain, swelling, stiffness, and difficulty moving the knee. Treatment may involve rest, physical therapy, or surgery depending on the severity.",
        "source": "medical"
    },
    {
        "question": "How can I prevent heatstroke?",
        "answer": "To prevent heatstroke, stay hydrated, avoid strenuous activity in hot weather, wear lightweight clothing, and take breaks in cool or shaded areas. Seek medical attention if symptoms like confusion or high body temperature occur.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn ACL?",
        "answer": "Symptoms of a torn ACL include a popping sound at the time of injury, severe pain, swelling, and instability in the knee. Treatment may involve physical therapy or surgery.",
        "source": "medical"
    },
    {
        "question": "How can I manage rosacea?",
        "answer": "To manage rosacea, avoid triggers like spicy foods and alcohol, use gentle skincare products, protect your skin from the sun, and consider prescribed medications or laser treatments.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a ruptured eardrum?",
        "answer": "Symptoms of a ruptured eardrum include ear pain, hearing loss, drainage from the ear, and ringing in the ear. Most ruptures heal on their own, but consult a doctor if symptoms persist.",
        "source": "medical"
    },
    {
        "question": "How can I prevent athlete's foot?",
        "answer": "To prevent athlete's foot, keep your feet clean and dry, wear breathable footwear, avoid walking barefoot in public areas, and use antifungal powders or sprays if prone to infections.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a herniated disc?",
        "answer": "Symptoms of a herniated disc include back pain, numbness or tingling in the legs, and muscle weakness. Treatment may involve physical therapy, medications, or surgery in severe cases.",
        "source": "medical"
    },
    {
        "question": "How can I manage psoriasis?",
        "answer": "To manage psoriasis, moisturize regularly, avoid triggers like stress and alcohol, use prescribed creams or light therapy, and consider systemic medications for severe cases.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn rotator cuff?",
        "answer": "Symptoms of a torn rotator cuff include shoulder pain, weakness, and difficulty lifting the arm. Treatment may involve physical therapy, rest, or surgery depending on the severity.",
        "source": "medical"
    },
    {
        "question": "How can I prevent ingrown toenails?",
        "answer": "To prevent ingrown toenails, trim nails straight across, wear properly fitting shoes, and avoid cutting nails too short. Soak the affected area in warm water if an ingrown toenail occurs.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a pulmonary embolism?",
        "answer": "Symptoms of a pulmonary embolism include sudden shortness of breath, chest pain, rapid heartbeat, and coughing up blood. Seek emergency medical attention if these symptoms occur.",
        "source": "medical"
    },
    {
        "question": "How can I manage gout?",
        "answer": "To manage gout, avoid purine-rich foods like red meat and shellfish, stay hydrated, take prescribed medications, and apply ice to affected joints during flare-ups.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a hiatal hernia?",
        "answer": "Symptoms of a hiatal hernia include heartburn, chest pain, difficulty swallowing, and regurgitation. Treatment may involve lifestyle changes, medications, or surgery in severe cases.",
        "source": "medical"
    },
    {
        "question": "How can I prevent dry skin?",
        "answer": "To prevent dry skin, moisturize regularly, avoid hot showers, use a humidifier, and drink plenty of water. Choose gentle, fragrance-free skincare products.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn ligament?",
        "answer": "Symptoms of a torn ligament include pain, swelling, bruising, and instability in the affected joint. Treatment may involve rest, physical therapy, or surgery depending on the severity.",
        "source": "medical"
    },
    {
        "question": "How can I manage tinnitus?",
        "answer": "To manage tinnitus, avoid loud noises, use white noise machines, reduce stress, and consider cognitive-behavioral therapy or hearing aids if symptoms are severe.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a kidney infection?",
        "answer": "Symptoms of a kidney infection include fever, chills, back or side pain, nausea, and frequent urination. Antibiotics are typically required for treatment.",
        "source": "medical"
    },
    {
        "question": "How can I prevent dandruff?",
        "answer": "To prevent dandruff, wash your hair regularly with an anti-dandruff shampoo, avoid excessive use of hair products, and manage stress. Consult a dermatologist if symptoms persist.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn Achilles tendon?",
        "answer": "Symptoms of a torn Achilles tendon include sudden pain, swelling, and difficulty walking. Surgery is often required for treatment.",
        "source": "medical"
    },
    {
        "question": "How can I manage chronic fatigue syndrome?",
        "answer": "To manage chronic fatigue syndrome, prioritize rest, maintain a balanced diet, engage in gentle exercise, and consider cognitive-behavioral therapy or medications for symptom relief.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a ruptured spleen?",
        "answer": "Symptoms of a ruptured spleen include severe abdominal pain, tenderness, and lightheadedness. Seek emergency medical attention if a ruptured spleen is suspected.",
        "source": "medical"
    },
    {
        "question": "How can I prevent bedsores?",
        "answer": "To prevent bedsores, change positions frequently, use supportive mattresses, keep skin clean and dry, and ensure proper nutrition and hydration.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn MCL?",
        "answer": "Symptoms of a torn MCL include knee pain, swelling, and instability. Treatment may involve rest, physical therapy, or surgery depending on the severity.",
        "source": "medical"
    },
    {
        "question": "How can I manage Raynaud's disease?",
        "answer": "To manage Raynaud's disease, keep hands and feet warm, avoid cold temperatures, reduce stress, and consider medications to improve blood flow.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a perforated eardrum?",
        "answer": "Symptoms of a perforated eardrum include ear pain, hearing loss, and drainage from the ear. Most perforations heal on their own, but consult a doctor if symptoms persist.",
        "source": "medical"
    },
    {
        "question": "How can I prevent jock itch?",
        "answer": "To prevent jock itch, keep the groin area clean and dry, wear breathable underwear, and use antifungal powders or creams if prone to infections.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn PCL?",
        "answer": "Symptoms of a torn PCL include knee pain, swelling, and difficulty walking. Treatment may involve rest, physical therapy, or surgery depending on the severity.",
        "source": "medical"
    },
    {
        "question": "How can I manage fibromyalgia?",
        "answer": "To manage fibromyalgia, prioritize sleep, engage in gentle exercise, manage stress, and consider medications or cognitive-behavioral therapy for symptom relief.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a ruptured appendix?",
        "answer": "Symptoms of a ruptured appendix include severe abdominal pain, fever, nausea, and vomiting. Seek emergency medical attention if a ruptured appendix is suspected.",
        "source": "medical"
    },
    {
        "question": "How can I prevent shingles?",
        "answer": "To prevent shingles, get the shingles vaccine, maintain a healthy immune system, and avoid close contact with individuals who have active shingles or chickenpox.",
        "source": "medical"
    },
    {
        "question": "What are the symptoms of a torn ACL?",
        "answer": "Symptoms of a torn ACL include a popping sound at the time of injury, severe pain, swelling, and instability in the knee. Treatment may involve physical therapy or surgery.",
        "source": "medical"
    },
    {
        "question": "What are the differential diagnoses for a patient presenting with chest pain?",
        "answer": "Chest pain can have multiple causes, including cardiac (e.g., myocardial infarction, angina, pericarditis), pulmonary (e.g., pulmonary embolism, pneumothorax), gastrointestinal (e.g., GERD, esophageal rupture), musculoskeletal (e.g., costochondritis), and psychological (e.g., panic attack). A thorough history, physical exam, and diagnostic tests (ECG, troponin levels, imaging) are essential for accurate diagnosis.",
        "source": "medical"
    },
    {
        "question": "How is sepsis diagnosed and managed in a clinical setting?",
        "answer": "Sepsis is diagnosed using the Sepsis-3 criteria: suspected infection plus an increase in SOFA score ≥2. Management includes: 1) Early administration of broad-spectrum antibiotics, 2) Fluid resuscitation, 3) Vasopressors if hypotensive, 4) Source control (e.g., draining abscesses), 5) Monitoring organ function and supportive care. Early intervention is critical to reduce mortality.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for rheumatoid arthritis?",
        "answer": "Treatment for rheumatoid arthritis (RA) includes: 1) Disease-modifying antirheumatic drugs (DMARDs) like methotrexate, 2) Biologic agents (e.g., TNF inhibitors), 3) Corticosteroids for acute flares, 4) NSAIDs for pain relief, 5) Physical therapy to maintain joint function. Treatment is tailored to disease severity and patient response.",
        "source": "medical"
    },
    {
        "question": "What are the key steps in managing acute stroke?",
        "answer": "Acute stroke management involves: 1) Rapid assessment using the NIH Stroke Scale, 2) Immediate imaging (CT or MRI) to rule out hemorrhage, 3) Administering thrombolytics (e.g., alteplase) within 4.5 hours of symptom onset if ischemic, 4) Mechanical thrombectomy for large vessel occlusion, 5) Monitoring in a stroke unit, and 6) Secondary prevention with antiplatelets, statins, and risk factor management.",
        "source": "medical"
    },
    {
        "question": "How is acute pancreatitis diagnosed and treated?",
        "answer": "Acute pancreatitis is diagnosed based on clinical symptoms (abdominal pain, nausea), elevated serum lipase/amylase, and imaging (CT or ultrasound). Treatment includes: 1) NPO (nothing by mouth) to rest the pancreas, 2) IV fluids for hydration, 3) Pain management, 4) Treating underlying causes (e.g., gallstones, alcohol cessation), and 5) Monitoring for complications like pseudocysts or infection.",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for systemic lupus erythematosus (SLE)?",
        "answer": "SLE is diagnosed using the 2019 EULAR/ACR criteria, which include clinical and immunological features such as malar rash, oral ulcers, arthritis, renal disorder, positive ANA, and specific antibodies (e.g., anti-dsDNA, anti-Smith). A score of ≥10 with at least one clinical criterion is required for diagnosis.",
        "source": "medical"
    },
    {
        "question": "How is diabetic ketoacidosis (DKA) managed in the emergency department?",
        "answer": "DKA management includes: 1) IV fluids (normal saline) for rehydration, 2) Insulin infusion to correct hyperglycemia and ketosis, 3) Electrolyte replacement (potassium, magnesium), 4) Monitoring blood glucose, electrolytes, and pH, 5) Treating underlying triggers (e.g., infection). Close monitoring in an ICU setting is often required.",
        "source": "medical"
    },
    {
        "question": "What are the indications for a liver transplant?",
        "answer": "Liver transplant is indicated for end-stage liver disease (e.g., cirrhosis, acute liver failure), hepatocellular carcinoma within Milan criteria, and certain metabolic disorders. Evaluation includes MELD score, absence of contraindications (e.g., active infection, severe cardiopulmonary disease), and psychosocial assessment.",
        "source": "medical"
    },
    {
        "question": "How is acute respiratory distress syndrome (ARDS) managed?",
        "answer": "ARDS management includes: 1) Mechanical ventilation with low tidal volumes (6 mL/kg ideal body weight), 2) Positive end-expiratory pressure (PEEP) to improve oxygenation, 3) Prone positioning in severe cases, 4) Treating the underlying cause (e.g., sepsis, pneumonia), 5) Fluid management to avoid overload, and 6) ECMO in refractory cases.",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for multiple sclerosis (MS)?",
        "answer": "MS is diagnosed using the McDonald criteria, which include clinical episodes of neurological dysfunction, MRI evidence of lesions in characteristic locations, and exclusion of alternative diagnoses. Cerebrospinal fluid analysis (oligoclonal bands) and evoked potentials may also support the diagnosis.",
        "source": "medical"
    },
    {
        "question": "How is acute kidney injury (AKI) managed?",
        "answer": "AKI management involves: 1) Identifying and treating the underlying cause (e.g., hypovolemia, nephrotoxins), 2) Optimizing fluid balance, 3) Avoiding nephrotoxic medications, 4) Monitoring electrolytes and acid-base status, 5) Renal replacement therapy (dialysis) in severe cases. Early intervention improves outcomes.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for metastatic breast cancer?",
        "answer": "Treatment for metastatic breast cancer depends on hormone receptor and HER2 status. Options include: 1) Hormonal therapy (e.g., tamoxifen, aromatase inhibitors), 2) Targeted therapy (e.g., trastuzumab for HER2+), 3) Chemotherapy, 4) Immunotherapy, 5) Palliative care for symptom management. Treatment is individualized based on patient factors.",
        "source": "medical"
    },
    {
        "question": "How is acute myocardial infarction (AMI) managed?",
        "answer": "AMI management includes: 1) Immediate aspirin and nitroglycerin, 2) Reperfusion therapy (PCI or thrombolysis), 3) Anticoagulants (e.g., heparin), 4) Beta-blockers and ACE inhibitors, 5) Pain management (e.g., morphine), 6) Monitoring for complications (e.g., arrhythmias, heart failure). Early reperfusion is critical.",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for Parkinson's disease?",
        "answer": "Parkinson's disease is diagnosed clinically based on the presence of bradykinesia plus at least one of the following: resting tremor, rigidity, or postural instability. Supportive features include asymmetric onset, response to levodopa, and absence of atypical features (e.g., early falls, rapid progression).",
        "source": "medical"
    },
    {
        "question": "How is acute asthma exacerbation managed in the emergency department?",
        "answer": "Acute asthma exacerbation management includes: 1) High-dose inhaled bronchodilators (e.g., albuterol), 2) Systemic corticosteroids, 3) Oxygen to maintain SpO2 >90%, 4) Magnesium sulfate in severe cases, 5) Monitoring for respiratory failure and considering mechanical ventilation if necessary.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for chronic obstructive pulmonary disease (COPD)?",
        "answer": "COPD treatment includes: 1) Bronchodilators (e.g., LABA, LAMA), 2) Inhaled corticosteroids for frequent exacerbations, 3) Pulmonary rehabilitation, 4) Oxygen therapy for severe hypoxemia, 5) Smoking cessation, 6) Vaccinations (e.g., influenza, pneumococcal). Severe cases may require surgical interventions like lung volume reduction.",
        "source": "medical"
    },
    {
        "question": "How is acute appendicitis diagnosed and treated?",
        "answer": "Acute appendicitis is diagnosed clinically (right lower quadrant pain, rebound tenderness) and confirmed with imaging (ultrasound or CT). Treatment is surgical appendectomy, with antibiotics (e.g., ceftriaxone, metronidazole) administered preoperatively. Non-operative management with antibiotics may be considered in uncomplicated cases.",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for Alzheimer's disease?",
        "answer": "Alzheimer's disease is diagnosed based on clinical criteria (e.g., memory impairment, cognitive decline) and exclusion of other causes. Biomarkers (e.g., amyloid PET, CSF tau) may support the diagnosis. Neuropsychological testing and imaging (MRI, CT) are used to assess severity and rule out other conditions.",
        "source": "medical"
    },
    {
        "question": "How is acute upper gastrointestinal bleeding managed?",
        "answer": "Management includes: 1) Stabilization (IV fluids, blood transfusion if needed), 2) Proton pump inhibitors (e.g., omeprazole), 3) Endoscopic intervention (e.g., banding, sclerotherapy), 4) Treating underlying causes (e.g., H. pylori, varices), 5) Monitoring for rebleeding and complications.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for metastatic melanoma?",
        "answer": "Treatment includes: 1) Immunotherapy (e.g., checkpoint inhibitors like pembrolizumab), 2) Targeted therapy (e.g., BRAF inhibitors for BRAF-mutated melanoma), 3) Chemotherapy, 4) Radiation for symptomatic lesions, 5) Surgical resection of isolated metastases. Treatment is tailored to mutation status and disease burden.",
        "source": "medical"
    },
    {
        "question": "How is acute bacterial meningitis diagnosed and treated?",
        "answer": "Diagnosis involves lumbar puncture (elevated WBC, low glucose, high protein) and CSF culture. Treatment includes: 1) Empiric antibiotics (e.g., ceftriaxone, vancomycin), 2) Dexamethasone to reduce inflammation, 3) Supportive care (e.g., fluids, antipyretics), 4) Isolation precautions until bacterial cause is confirmed.",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for bipolar disorder?",
        "answer": "Bipolar disorder is diagnosed based on DSM-5 criteria: episodes of mania (elevated mood, increased activity) and depression (low mood, anhedonia). Hypomanic episodes may also occur. Diagnosis requires ruling out other causes (e.g., substance use, medical conditions).",
        "source": "medical"
    },
    {
        "question": "How is acute pulmonary embolism (PE) managed?",
        "answer": "PE management includes: 1) Anticoagulation (e.g., heparin, DOACs), 2) Thrombolysis for massive PE, 3) Oxygen therapy, 4) Hemodynamic support, 5) Inferior vena cava filter in contraindications to anticoagulation. Risk stratification (e.g., PESI score) guides treatment intensity.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for chronic hepatitis C?",
        "answer": "Treatment involves direct-acting antivirals (DAAs) like sofosbuvir/ledipasvir or glecaprevir/pibrentasvir, tailored to genotype and liver function. Cure rates exceed 95%. Monitoring for liver fibrosis and hepatocellular carcinoma is essential post-treatment.",
        "source": "medical"
    },
    {
        "question": "How is acute diverticulitis managed?",
        "answer": "Management includes: 1) Antibiotics (e.g., ciprofloxacin, metronidazole), 2) Bowel rest (clear liquids), 3) Pain management, 4) Hospitalization for severe cases (e.g., abscess, perforation), 5) Surgical intervention for complications (e.g., fistula, obstruction).",
        "source": "medical"
    },
    {
        "question": "What are the diagnostic criteria for Crohn's disease?",
        "answer": "Crohn's disease is diagnosed based on clinical symptoms (diarrhea, abdominal pain, weight loss), endoscopic findings (patchy inflammation, cobblestoning), imaging (CT/MRI), and histology (transmural inflammation, granulomas). Differential diagnosis includes ulcerative colitis and infectious colitis.",
        "source": "medical"
    },
    {
        "question": "How is acute cholecystitis managed?",
        "answer": "Management includes: 1) NPO, 2) IV fluids, 3) Antibiotics (e.g., piperacillin-tazobactam), 4) Pain control, 5) Cholecystectomy (laparoscopic or open). Percutaneous cholecystostomy may be used in high-risk patients.",
        "source": "medical"
    },
    {
        "question": "What are the treatment options for advanced prostate cancer?",
        "answer": "Treatment includes: 1) Androgen deprivation therapy (e.g., leuprolide), 2) Chemotherapy (e.g., docetaxel), 3) Targeted therapy (e.g., abiraterone), 4) Radiation for symptomatic metastases, 5) Bone-targeted therapy (e.g., denosumab) for skeletal-related events.",
        "source": "medical"
    },
    {
        "question": "How is acute hypertensive crisis managed?",
        "answer": "Management includes: 1) IV antihypertensives (e.g., labetalol, nicardipine) for hypertensive emergency, 2) Oral agents (e.g., nifedipine, clonidine) for urgency, 3) Monitoring for end-organ damage (e.g., encephalopathy, renal failure), 4) Gradual BP reduction to avoid ischemia.",
        "source": "medical"
    },
    {
        "question": "Hello doctor, I'm really scared! What should I do if I experience sudden chest pain? 😟",
        "answer": "If you have sudden chest pain: 1) Call emergency services immediately, 2) Chew 325 mg of aspirin if not allergic, 3) Avoid physical exertion, 4) If prescribed, take nitroglycerin sublingually, 5) Stay calm and monitor for symptoms like shortness of breath or dizziness.",
        "source": "medical"
    },
    {
        "question": "Hi doctor, I feel so exhausted all the time. Could it be anemia? 😞",
        "answer": "Anemia may cause fatigue. Evaluation includes: 1) CBC to check hemoglobin levels, 2) Iron studies for iron deficiency, 3) Vitamin B12 and folate levels, 4) Bone marrow biopsy in unexplained cases, 5) Treatment depends on the cause (e.g., iron supplements, B12 injections).",
        "source": "medical"
    },
    {
        "question": "Good morning, doctor! I'm worried about my high blood sugar. What's the best way to control diabetes? 😊",
        "answer": "Managing diabetes includes: 1) Lifestyle changes (healthy diet, regular exercise), 2) Oral medications (e.g., metformin, sulfonylureas), 3) Insulin therapy for type 1 or severe type 2 diabetes, 4) Regular glucose monitoring, 5) Managing complications (e.g., foot care, eye exams).",
        "source": "medical"
    },
    {
        "question": "Doctor, I'm so anxious! My child has a fever. What should I do? 😰",
        "answer": "For fever in children: 1) Give antipyretics like acetaminophen or ibuprofen (age-appropriate dose), 2) Ensure hydration with fluids, 3) Dress them lightly to prevent overheating, 4) Seek medical help if fever >102°F, difficulty breathing, or persistent symptoms.",
        "source": "medical"
    },
    {
        "question": "Hey doc, I'm feeling really down lately… Could this be depression? 😢",
        "answer": "Depression signs include: 1) Persistent sadness, loss of interest, 2) Sleep and appetite changes, 3) Fatigue, difficulty concentrating, 4) Suicidal thoughts (seek immediate help if present). Treatment includes therapy, medications (SSRIs, SNRIs), and lifestyle modifications.",
        "source": "medical"
    },
    {
        "question": "Hello doctor! I'm excited but also nervous—what should I do to have a healthy pregnancy? 🤰💖",
        "answer": "Healthy pregnancy tips: 1) Take prenatal vitamins (folic acid, iron), 2) Attend regular prenatal checkups, 3) Maintain a balanced diet, 4) Stay active but avoid high-impact exercises, 5) Avoid alcohol, smoking, and excessive caffeine, 6) Manage stress and get enough rest.",
        "source": "medical"
    },
    {
        "question": "Good evening, doctor! I can't sleep at all, and it's making me so frustrated. What could be causing this? 😩",
        "answer": "Insomnia can be caused by: 1) Stress and anxiety, 2) Poor sleep hygiene (caffeine, screen exposure before bed), 3) Sleep disorders (e.g., sleep apnea, restless legs syndrome), 4) Medical conditions (e.g., hyperthyroidism), 5) Medications. Treatment includes cognitive behavioral therapy, relaxation techniques, and, if needed, short-term medication use.",
        "source": "medical"
    },
    {
        "question": "Doctor, I'm scared! My friend is struggling with drug addiction. How can they get help? 😟",
        "answer": "Addiction treatment includes: 1) Detox programs under medical supervision, 2) Medication-assisted treatment (e.g., methadone, buprenorphine for opioid addiction), 3) Cognitive-behavioral therapy (CBT), 4) Support groups (e.g., NA, AA), 5) Lifestyle modifications and relapse prevention strategies.",
        "source": "medical"
    },
    {
        "question": "Hello doctor! I've heard of personalized medicine—how does it work? 🤔",
        "answer": "Personalized medicine tailors treatments based on: 1) Genetic profile (e.g., pharmacogenomics to predict drug response), 2) Biomarkers (e.g., HER2 for breast cancer therapy), 3) Lifestyle and environmental factors, 4) AI-driven health data analysis, 5) Precision therapies for targeted effectiveness.",
        "source": "medical"
    },
    {
        "question": "Hey doc, I'm concerned about healthcare fairness. How can we ensure health diversity? 🤷🏽‍♂️",
        "answer": "Ensuring health diversity includes: 1) Addressing racial and socioeconomic disparities, 2) Improving access to quality healthcare, 3) Culturally competent medical training, 4) Encouraging diversity in clinical trials, 5) Using technology to reach underserved communities.",
        "source": "medical"
    },
    {
        "question": "Doctor, I keep hearing about electronic health records (EHRs). Are they really helpful? 📋",
        "answer": "EHRs improve healthcare by: 1) Enhancing communication among providers, 2) Reducing medical errors, 3) Streamlining patient history access, 4) Enabling telemedicine integration, 5) Protecting data with security protocols while addressing privacy concerns.",
        "source": "medical"
    },
    {
        "question": "Hi doctor! I'm fascinated by genetic engineering. How is it being used in medicine? 🧬",
        "answer": "Genetic engineering is used for: 1) Gene therapy to treat genetic disorders (e.g., sickle cell anemia), 2) CRISPR for precise gene editing, 3) Developing genetically modified organs for transplants, 4) Cancer treatment via engineered immune cells (CAR-T therapy), 5) Personalized medicine applications.",
        "source": "medical"
    },
    {
        "question": "Hello doctor! I want a more natural approach to healthcare. What is integrative medicine? 🌿",
        "answer": "Integrative medicine combines: 1) Conventional treatments (e.g., medications, surgery), 2) Complementary therapies (e.g., acupuncture, herbal medicine), 3) Mind-body practices (e.g., yoga, meditation), 4) Nutritional and lifestyle changes, 5) Evidence-based holistic care for overall well-being.",
        "source": "medical"
    },
    {
        "question": "Hey doc, I'm curious about medical marijuana. Is it really effective for treatment? 🌱",
        "answer": "Medical marijuana can help with: 1) Chronic pain management, 2) Nausea and vomiting from chemotherapy, 3) Seizure disorders (e.g., epilepsy), 4) Muscle spasticity in multiple sclerosis, 5) Anxiety and PTSD (under medical supervision). Legal status varies by region.",
        "source": "medical"
    },
    {
        "question": "Doctor, I feel so overwhelmed lately… Could this be a mental health issue? 😢",
        "answer": "Mental health concerns include: 1) Anxiety disorders (excessive worry, panic attacks), 2) Depression (persistent sadness, loss of interest), 3) PTSD (flashbacks, nightmares), 4) Bipolar disorder (mood swings), 5) Seeking professional therapy, medication, and lifestyle changes for management.",
        "source": "medical"
    },
    {
        "question": "Good morning, doctor! I want to eat healthier. What are the basics of good nutrition? 🥗",
        "answer": "Good nutrition includes: 1) Eating a balanced diet rich in fruits, vegetables, and whole grains, 2) Consuming lean proteins and healthy fats, 3) Limiting processed foods and added sugars, 4) Staying hydrated, 5) Maintaining portion control to support a healthy weight.",
        "source": "medical"
    },
    {
        "question": "Doctor, my back pain is unbearable! What are my treatment options? 😖",
        "answer": "Pain management includes: 1) Physical therapy and stretching exercises, 2) Medications (NSAIDs, muscle relaxants), 3) Interventional procedures (e.g., nerve blocks, epidural injections), 4) Alternative treatments (acupuncture, chiropractic care), 5) Surgery for severe cases.",
        "source": "medical"
    },
    {
        "question": "Doctor, my grandmother is in palliative care. How can I support her? 😔",
        "answer": "Supporting a loved one in palliative care includes: 1) Ensuring comfort with pain management, 2) Providing emotional support and companionship, 3) Respecting their wishes and advance directives, 4) Seeking hospice services if needed, 5) Practicing self-care as a caregiver.",
        "source": "medical"
    },
    {
        "question": "Hi doctor! I just found out I'm pregnant. What should I do first? 🤰💖",
        "answer": "For a healthy pregnancy: 1) Schedule your first prenatal visit, 2) Start taking prenatal vitamins (folic acid, iron), 3) Maintain a balanced diet and hydration, 4) Avoid smoking, alcohol, and harmful substances, 5) Manage stress and get adequate rest.",
        "source": "medical"
    },
    {
        "question": "Hey doctor! I can't make it to the clinic. How does telehealth work? 💻",
        "answer": "Telehealth provides remote healthcare through: 1) Virtual consultations via video or phone, 2) Remote monitoring of chronic conditions, 3) E-prescriptions for medications, 4) AI-driven symptom checkers, 5) Increased accessibility for rural and underserved communities.",
        "source": "medical"
    },
    {
        "question": "Hi doctor, I'm really worried about my constant headaches. Could it be something serious? 😟",
        "answer": "Frequent headaches can have various causes: 1) Tension headaches due to stress, 2) Migraines characterized by throbbing pain, 3) Sinus issues causing facial pressure, 4) Eye strain from prolonged screen time, 5) Rarely, more serious conditions like brain tumors. It's important to consult a healthcare provider for proper evaluation and management.",
        "source": "medical"
    },
    {
        "question": "Hello doctor, I'm feeling anxious about my upcoming surgery. How can I prepare myself? 😰",
        "answer": "Pre-surgery preparation includes: 1) Discussing concerns with your surgeon, 2) Following pre-operative instructions (e.g., fasting), 3) Arranging post-operative care and support, 4) Practicing relaxation techniques to manage anxiety, 5) Ensuring all medical information is up-to-date. Open communication with your healthcare team is key.",
        "source": "medical"
    },
    {
        "question": "Good morning, doctor! I'm thrilled to start a new fitness routine. Any tips to do it safely? 😊",
        "answer": "Starting a fitness routine safely involves: 1) Consulting with a healthcare provider, especially if you have existing health conditions, 2) Setting realistic goals, 3) Beginning with low-impact exercises, 4) Incorporating rest days to prevent overuse injuries, 5) Staying hydrated and maintaining a balanced diet.",
        "source": "medical"
    },
    {
        "question": "Hey doc, I'm feeling down because of my recent weight gain. What can I do to manage it? 😞",
        "answer": "Managing weight gain includes: 1) Evaluating dietary habits and reducing calorie-dense foods, 2) Increasing physical activity with both aerobic and strength-training exercises, 3) Seeking support from a nutritionist or weight management program, 4) Monitoring portion sizes, 5) Addressing emotional factors that may contribute to overeating.",
        "source": "medical"
    },
    {
        "question": "Doctor, I'm scared about the recent outbreak of infectious diseases. How can I protect myself? 😷",
        "answer": "Protecting yourself from infectious diseases involves: 1) Practicing good hand hygiene, 2) Staying up-to-date with vaccinations, 3) Avoiding close contact with sick individuals, 4) Wearing masks in crowded or high-risk areas, 5) Following public health guidelines and staying informed about outbreaks.",
        "source": "medical"
    },
    {
"question": "Hi, I've been feeling really anxious lately. Can you help me understand what might be going on?",
"answer": "Of course, I'm here to help. Anxiety can be caused by a variety of factors, including stress, hormonal imbalances, or underlying mental health conditions like generalized anxiety disorder. It's important to talk to a healthcare provider who can evaluate your symptoms and recommend treatments such as therapy, lifestyle changes, or medications if needed. How long have you been feeling this way?",
"source": "medical",
"emotional_tone": "supportive"
},
{
"question": "What should I do if I think I'm having a heart attack?",
"answer": "If you suspect a heart attack, it's crucial to act immediately. Call emergency services right away. While waiting, chew an aspirin (if not allergic) to help thin your blood, sit or lie down, and try to stay calm. Symptoms like chest pain, shortness of breath, or radiating pain in the arm or jaw should never be ignored. You're not alone—help is on the way.",
"source": "medical",
"emotional_tone": "urgent and reassuring"
},
{
"question": "Hello, my child has a high fever. What should I do?",
"answer": "Hi there, I understand how worrying it can be when your child is unwell. For a high fever, you can start by giving them acetaminophen or ibuprofen (based on their age and weight) to reduce the fever. Make sure they stay hydrated and dress them in light clothing. If the fever persists for more than 48 hours, is accompanied by a rash, or if your child seems unusually lethargic, seek medical attention right away. How is your child feeling otherwise?",
"source": "medical",
"emotional_tone": "caring and empathetic"
},
{
"question": "I've been diagnosed with diabetes. What lifestyle changes should I make?",
"answer": "I know a diabetes diagnosis can feel overwhelming, but you're taking the right step by seeking information. Key lifestyle changes include: 1) Eating a balanced diet rich in whole grains, vegetables, and lean proteins, 2) Regular exercise (e.g., 30 minutes of walking daily), 3) Monitoring blood sugar levels, and 4) Avoiding sugary drinks and processed foods. Your healthcare provider can help you create a personalized plan. You've got this!",
"source": "medical",
"emotional_tone": "encouraging"
},
{
"question": "What are the signs of a stroke, and what should I do if I notice them?",
"answer": "The signs of a stroke can be remembered with the acronym FAST: Face drooping, Arm weakness, Speech difficulty, and Time to call emergency services. Other symptoms include sudden confusion, trouble seeing, or severe headache. If you notice these signs, don't wait—call for help immediately. Quick action can save a life. You're doing the right thing by being aware.",
"source": "medical",
"emotional_tone": "urgent and informative"
},
{
"question": "I've been feeling really tired all the time. Could it be something serious?",
"answer": "I'm sorry you're feeling this way—fatigue can be really draining. It could be due to a number of reasons, such as anemia, thyroid issues, sleep disorders, or even stress and depression. It's important to see a doctor who can run some tests and help pinpoint the cause. In the meantime, try to get enough rest, eat nutritious meals, and stay hydrated. How long has this been going on?",
"source": "medical",
"emotional_tone": "compassionate"
},
{
"question": "What are the treatment options for migraines?",
"answer": "Migraines can be debilitating, but there are ways to manage them. Treatment options include: 1) Acute medications like triptans or NSAIDs for relief during an attack, 2) Preventive medications (e.g., beta-blockers, anticonvulsants) if migraines are frequent, 3) Lifestyle changes like stress management and regular sleep, and 4) Avoiding triggers such as certain foods or bright lights. A neurologist can help tailor a plan for you. You're not alone in this.",
"source": "medical",
"emotional_tone": "supportive"
},
{
"question": "How can I support a loved one who's been diagnosed with cancer?",
"answer": "It's wonderful that you want to support your loved one—they're lucky to have you. You can help by: 1) Listening without judgment, 2) Offering practical help like driving them to appointments, 3) Educating yourself about their condition, and 4) Being patient as they navigate their emotions. Remember to take care of yourself too, as caregiving can be emotionally taxing. You're doing a great job.",
"source": "medical",
"emotional_tone": "empathetic and encouraging"
},
{
"question": "What are the symptoms of depression, and when should I seek help?",
"answer": "Depression can feel isolating, but it's important to know you're not alone. Symptoms include persistent sadness, loss of interest in activities, changes in appetite or sleep, fatigue, and feelings of worthlessness. If these symptoms last for more than two weeks and interfere with daily life, it's time to seek help from a mental health professional. Treatment options like therapy and medication can make a big difference. You deserve support.",
"source": "medical",
"emotional_tone": "compassionate"
},
{
"question": "How can I manage chronic pain without relying too much on medications?",
"answer": "Chronic pain can be challenging, but there are ways to manage it holistically. Consider: 1) Physical therapy or gentle exercises like yoga, 2) Mindfulness and relaxation techniques, 3) Acupuncture or massage therapy, 4) Cognitive-behavioral therapy to address the emotional impact of pain, and 5) A healthy diet to reduce inflammation. Always discuss these options with your doctor to create a safe and effective plan. You're stronger than you think.",
"source": "medical",
"emotional_tone": "encouraging"
},
{
        "question": "Hi, I've been having trouble sleeping lately. It's really affecting my mood. What can I do?",
        "answer": "I'm sorry to hear you're struggling with sleep—it can really take a toll on your well-being. You might try establishing a bedtime routine, avoiding screens before bed, and creating a calm sleep environment. If this persists, it could be helpful to see a doctor to rule out conditions like insomnia or sleep apnea. How long has this been going on?",
        "source": "medical",
        "emotional_tone": "caring"
    },
    {
        "question": "Hello, I've been feeling a sharp pain in my lower back. Should I be worried?",
        "answer": "Hi there, back pain can be really uncomfortable. Sharp pain could be due to muscle strain, a herniated disc, or even kidney issues. If the pain is severe, persistent, or accompanied by fever or numbness, it's best to see a doctor. In the meantime, rest, apply heat or ice, and avoid heavy lifting. How long have you had this pain?",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hey, I've been feeling really down and unmotivated lately. Could this be depression?",
        "answer": "I'm really sorry you're feeling this way—it sounds tough. Persistent sadness and lack of motivation can be signs of depression, especially if they last for more than two weeks. It's important to talk to a mental health professional who can help you explore treatment options like therapy or medication. You're not alone in this, and help is available.",
        "source": "medical",
        "emotional_tone": "empathetic"
    },
    {
        "question": "Hi, my partner has been diagnosed with Alzheimer's. How can I best support them?",
        "answer": "Hi, I can only imagine how challenging this must be for you. Supporting a loved one with Alzheimer's involves patience, understanding, and practical help. You can create a safe environment, establish routines, and use reminders for daily tasks. Joining a support group for caregivers can also be incredibly helpful. Remember to take care of yourself too—you're doing an amazing job.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hello, I've been experiencing chest tightness and shortness of breath. Should I be concerned?",
        "answer": "Hi, chest tightness and shortness of breath can be concerning, especially if they come on suddenly. These symptoms could indicate anything from anxiety to a heart or lung condition. It's important to seek medical attention right away to rule out serious issues like a heart attack or pulmonary embolism. Please don't wait—your health is important.",
        "source": "medical",
        "emotional_tone": "urgent and caring"
    },
    {
        "question": "Hey, I've been having frequent headaches. What could be causing them?",
        "answer": "Hi there, frequent headaches can be really frustrating. They could be caused by stress, dehydration, poor posture, or even underlying conditions like migraines or high blood pressure. Keeping a headache diary to track triggers and patterns can help. If they persist or worsen, it's a good idea to see a doctor. How long have you been experiencing this?",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi, I've been feeling really fatigued even after a full night's sleep. What could be wrong?",
        "answer": "Hi, I'm sorry you're feeling this way—fatigue can be really draining. It could be due to conditions like anemia, thyroid issues, sleep apnea, or even chronic fatigue syndrome. A doctor can run tests to help identify the cause. In the meantime, try to maintain a healthy diet, stay hydrated, and get regular exercise. How long has this been happening?",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hello, I've been having trouble managing my stress. Do you have any tips?",
        "answer": "Hi, stress can feel overwhelming, but there are ways to manage it. You might try mindfulness practices like meditation or deep breathing, regular exercise, and setting aside time for hobbies you enjoy. If stress feels unmanageable, talking to a therapist can be incredibly helpful. You're not alone, and it's okay to ask for help.",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Hey, I've been having stomach pain and bloating after eating. What could this be?",
        "answer": "Hi there, stomach pain and bloating can be uncomfortable and might be due to food intolerances, irritable bowel syndrome (IBS), or other digestive issues. Keeping a food diary to track what you eat and how you feel afterward can help identify triggers. If symptoms persist, it's a good idea to see a gastroenterologist. How long have you been experiencing this?",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi, I've been having trouble concentrating and feel like my memory is slipping. Should I be worried?",
        "answer": "Hi there, I understand how concerning that must feel. Trouble concentrating and memory issues can be caused by stress, lack of sleep, or even conditions like ADHD or thyroid problems. It's a good idea to see a doctor who can evaluate your symptoms and recommend tests if needed. In the meantime, try to get enough rest and reduce stress. You're not alone in this.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hello, I've been feeling a constant ringing in my ears. What could this be?",
        "answer": "Hi, that sounds really frustrating. Ringing in the ears, known as tinnitus, can be caused by exposure to loud noises, ear infections, or even stress. It's important to see an ENT specialist to rule out underlying issues. In the meantime, avoiding loud noises and managing stress might help. How long have you been experiencing this?",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hey, I've been having frequent heartburn. What can I do to manage it?",
        "answer": "Hi there, frequent heartburn can be really uncomfortable. You might try avoiding trigger foods like spicy or fatty meals, eating smaller portions, and not lying down right after eating. Over-the-counter antacids can help, but if symptoms persist, it's best to see a doctor to rule out conditions like GERD. How long has this been going on?",
        "source": "medical",
        "emotional_tone": "caring"
    },
    {
        "question": "Hi, I've been feeling really dizzy lately, especially when I stand up. What could be causing this?",
        "answer": "Hi, I'm sorry you're feeling this way—dizziness can be unsettling. It could be due to dehydration, low blood pressure, or even an inner ear issue. Make sure you're drinking enough water and standing up slowly. If the dizziness persists or worsens, it's important to see a doctor to rule out serious causes. You're doing the right thing by seeking help.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello, I've been having joint pain and stiffness, especially in the mornings. Could this be arthritis?",
        "answer": "Hi there, joint pain and stiffness, particularly in the mornings, can be signs of arthritis, such as rheumatoid arthritis or osteoarthritis. It's important to see a rheumatologist who can run tests and recommend treatments like medications or physical therapy. In the meantime, gentle exercises and warm compresses might help. How long have you been experiencing this?",
        "source": "medical",
        "emotional_tone": "empathetic"
    },
    {
        "question": "Hey, I've been feeling really irritable and moody lately. Could this be hormonal?",
        "answer": "Hi, I understand how frustrating that can be. Irritability and mood swings can indeed be linked to hormonal changes, such as those during menstruation, pregnancy, or menopause. Stress and lack of sleep can also play a role. If this persists, it's a good idea to see a doctor who can check your hormone levels and suggest ways to manage it. You're not alone in this.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi, I've been having trouble breathing through my nose. What could be causing this?",
        "answer": "Hi there, nasal congestion can be really uncomfortable. It could be due to allergies, a sinus infection, or even a deviated septum. Over-the-counter nasal sprays or antihistamines might help, but if it persists, it's best to see an ENT specialist. How long have you been experiencing this?",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello, I've been feeling a lump in my throat when I swallow. Should I be worried?",
        "answer": "Hi, I understand how concerning that must feel. A lump in the throat, known as globus sensation, can be caused by acid reflux, stress, or even muscle tension. It's a good idea to see a doctor to rule out any serious issues. In the meantime, staying hydrated and managing stress might help. You're not alone in this.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hey, I've been having frequent urination and feel really thirsty all the time. What could this mean?",
        "answer": "Hi there, those symptoms could be signs of diabetes or a urinary tract infection. It's important to see a doctor who can check your blood sugar levels and run tests to determine the cause. In the meantime, stay hydrated and avoid sugary drinks. You're doing the right thing by seeking help.",
        "source": "medical",
        "emotional_tone": "urgent and caring"
    },
    {
        "question": "Hi, I've been feeling really weak and tired, even though I'm eating well. What could be wrong?",
        "answer": "Hi, I'm sorry you're feeling this way—weakness and fatigue can be really draining. It could be due to anemia, vitamin deficiencies, or even chronic conditions like hypothyroidism. A doctor can run tests to identify the cause and recommend treatments. In the meantime, try to get enough rest and eat iron-rich foods. You're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Good morning, doctor. I'm feeling anxious about my upcoming surgery. Is this normal? 😟",
        "answer": "It's completely normal to feel anxious before surgery. To manage this anxiety: 1) Discuss your concerns with your surgical team, 2) Practice relaxation techniques like deep breathing or meditation, 3) Ensure you understand the procedure and recovery process, 4) Consider speaking with a counselor if anxiety persists. Remember, your feelings are valid, and support is available.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hi doctor, I'm excited to start a new exercise routine but worried about potential injuries. Any advice? 😊",
        "answer": "It's great that you're enthusiastic about exercising! To minimize injury risk: 1) Start slowly and gradually increase intensity, 2) Warm up before and cool down after workouts, 3) Use proper form and equipment, 4) Listen to your body and rest when needed, 5) Stay hydrated and maintain a balanced diet. Enjoy your fitness journey!",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Hello doctor, I'm concerned about my recent forgetfulness. Could it be something serious? 😕",
        "answer": "Experiencing forgetfulness can be unsettling. Possible causes include: 1) Stress or anxiety, 2) Sleep deprivation, 3) Nutritional deficiencies, 4) Medication side effects, 5) Underlying medical conditions like thyroid issues or, in rare cases, early dementia. It's important to consult with a healthcare provider for a thorough evaluation. You're taking a proactive step by seeking help.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm thrilled to be pregnant but also nervous about what's ahead. What should I expect? 🤰😊",
        "answer": "Congratulations on your pregnancy! It's natural to feel a mix of excitement and nervousness. During this time: 1) Schedule regular prenatal visits, 2) Maintain a balanced diet rich in essential nutrients, 3) Engage in safe, moderate exercise, 4) Educate yourself about the stages of pregnancy and childbirth, 5) Reach out to support groups or classes for expectant parents. Embrace this journey, and remember, support is available every step of the way.",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Good afternoon, doctor. I'm feeling overwhelmed by my recent diagnosis. How can I cope? 😔",
        "answer": "Receiving a new diagnosis can be overwhelming. To cope: 1) Gather information about your condition to understand it better, 2) Connect with support groups or others facing similar challenges, 3) Consider speaking with a mental health professional, 4) Practice self-care activities that bring you comfort, 5) Lean on friends and family for support. Remember, it's okay to seek help, and you're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
     {
        "question": "Good morning, doctor. I've been experiencing frequent heartburn lately. Should I be concerned? 😕",
        "answer": "Frequent heartburn can be uncomfortable and may indicate gastroesophageal reflux disease (GERD). It's important to monitor your symptoms and consult with a healthcare provider for proper evaluation and management.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling anxious about my upcoming surgery. Is this normal? 😟",
        "answer": "It's completely normal to feel anxious before surgery. To manage this anxiety: 1) Discuss your concerns with your surgical team, 2) Practice relaxation techniques like deep breathing or meditation, 3) Ensure you understand the procedure and recovery process, 4) Consider speaking with a counselor if anxiety persists. Remember, your feelings are valid, and support is available.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hello doctor, I'm concerned about my recent forgetfulness. Could it be something serious? 😕",
        "answer": "Experiencing forgetfulness can be unsettling. Possible causes include: 1) Stress or anxiety, 2) Sleep deprivation, 3) Nutritional deficiencies, 4) Medication side effects, 5) Underlying medical conditions like thyroid issues or, in rare cases, early dementia. It's important to consult with a healthcare provider for a thorough evaluation. You're taking a proactive step by seeking help.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm thrilled to be pregnant but also nervous about what's ahead. What should I expect? 🤰😊",
        "answer": "Congratulations on your pregnancy! It's natural to feel a mix of excitement and nervousness. During this time: 1) Schedule regular prenatal visits, 2) Maintain a balanced diet rich in essential nutrients, 3) Engage in safe, moderate exercise, 4) Educate yourself about the stages of pregnancy and childbirth, 5) Reach out to support groups or classes for expectant parents. Embrace this journey, and remember, support is available every step of the way.",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Good afternoon, doctor. I'm feeling overwhelmed by my recent diagnosis. How can I cope? 😔",
        "answer": "Receiving a new diagnosis can be overwhelming. To cope: 1) Gather information about your condition to understand it better, 2) Connect with support groups or others facing similar challenges, 3) Consider speaking with a mental health professional, 4) Practice self-care activities that bring you comfort, 5) Lean on friends and family for support. Remember, it's okay to seek help, and you're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hi doctor, I'm worried about my recent weight loss. What could be causing it? 😟",
        "answer": "Unintentional weight loss can be concerning. Potential causes include: 1) Thyroid disorders, 2) Diabetes, 3) Gastrointestinal issues, 4) Cancer, 5) Chronic infections. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm feeling constantly fatigued. What might be the reason? 😴",
        "answer": "Persistent fatigue can result from various factors, such as: 1) Anemia, 2) Sleep disorders like sleep apnea, 3) Chronic fatigue syndrome, 4) Depression or anxiety, 5) Nutritional deficiencies. A comprehensive evaluation by a healthcare provider can help identify the cause and guide appropriate treatment.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Good morning, doctor. I'm experiencing shortness of breath lately. Should I be worried? 😰",
        "answer": "Shortness of breath can be caused by various conditions, including: 1) Asthma, 2) Chronic obstructive pulmonary disease (COPD), 3) Heart problems, 4) Anxiety disorders, 5) Anemia. It's important to consult with a healthcare provider to determine the cause and receive appropriate treatment.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling dizzy quite often. What could be the reason? 😵",
        "answer": "Frequent dizziness can result from several factors, such as: 1) Inner ear issues, 2) Low blood pressure, 3) Dehydration, 4) Medication side effects, 5) Neurological conditions. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm experiencing chest pain. Should I be alarmed? 😟",
        "answer": "Chest pain can have various causes, some of which are serious. Potential causes include: 1) Heart-related issues like angina or heart attack, 2) Gastroesophageal reflux disease (GERD), 3) Muscle strain, 4) Anxiety or panic attacks, 5) Lung conditions. It's crucial to seek immediate medical attention to determine the cause and receive appropriate care.",
        "source": "medical",
        "emotional_tone": "urgent"
    },
    {
        "question": "Good afternoon, doctor. I'm having trouble sleeping lately. What can I do? 😴",
        "answer": "Difficulty sleeping can be addressed by: 1) Establishing a regular sleep schedule, 2) Creating a comfortable sleep environment, 3) Limiting caffeine and alcohol intake, 4) Avoiding screens before bedtime, 5) Practicing relaxation techniques. If sleep problems persist, consult with a healthcare provider for further evaluation.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm feeling unusually thirsty all the time. What could be causing this? 😕",
        "answer": "Excessive thirst can be a symptom of various conditions, including: 1) Diabetes mellitus, 2) Dehydration, 3) Diabetes insipidus, 4) Kidney problems, 5) Medication side effects. It's important to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
     {
        "question": "Good morning, doctor. I've been experiencing frequent urination lately. Should I be concerned? 😕",
        "answer": "Frequent urination can result from various causes, such as urinary tract infections, diabetes, or prostate issues. It's important to consult with a healthcare provider for a proper evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling anxious about my upcoming surgery. Is this normal? 😟",
        "answer": "It's completely normal to feel anxious before surgery. To manage this anxiety: 1) Discuss your concerns with your surgical team, 2) Practice relaxation techniques like deep breathing or meditation, 3) Ensure you understand the procedure and recovery process, 4) Consider speaking with a counselor if anxiety persists. Remember, your feelings are valid, and support is available.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hello doctor, I'm concerned about my recent forgetfulness. Could it be something serious? 😕",
        "answer": "Experiencing forgetfulness can be unsettling. Possible causes include: 1) Stress or anxiety, 2) Sleep deprivation, 3) Nutritional deficiencies, 4) Medication side effects, 5) Underlying medical conditions like thyroid issues or, in rare cases, early dementia. It's important to consult with a healthcare provider for a thorough evaluation. You're taking a proactive step by seeking help.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm thrilled to be pregnant but also nervous about what's ahead. What should I expect? 🤰😊",
        "answer": "Congratulations on your pregnancy! It's natural to feel a mix of excitement and nervousness. During this time: 1) Schedule regular prenatal visits, 2) Maintain a balanced diet rich in essential nutrients, 3) Engage in safe, moderate exercise, 4) Educate yourself about the stages of pregnancy and childbirth, 5) Reach out to support groups or classes for expectant parents. Embrace this journey, and remember, support is available every step of the way.",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Good afternoon, doctor. I'm feeling overwhelmed by my recent diagnosis. How can I cope? 😔",
        "answer": "Receiving a new diagnosis can be overwhelming. To cope: 1) Gather information about your condition to understand it better, 2) Connect with support groups or others facing similar challenges, 3) Consider speaking with a mental health professional, 4) Practice self-care activities that bring you comfort, 5) Lean on friends and family for support. Remember, it's okay to seek help, and you're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hi doctor, I'm worried about my recent weight loss. What could be causing it? 😟",
        "answer": "Unintentional weight loss can be concerning. Potential causes include: 1) Thyroid disorders, 2) Diabetes, 3) Gastrointestinal issues, 4) Cancer, 5) Chronic infections. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm feeling constantly fatigued. What might be the reason? 😴",
        "answer": "Persistent fatigue can result from various factors, such as: 1) Anemia, 2) Sleep disorders like sleep apnea, 3) Chronic fatigue syndrome, 4) Depression or anxiety, 5) Nutritional deficiencies. A comprehensive evaluation by a healthcare provider can help identify the cause and guide appropriate treatment.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Good morning, doctor. I'm experiencing shortness of breath lately. Should I be worried? 😰",
        "answer": "Shortness of breath can be caused by various conditions, including: 1) Asthma, 2) Chronic obstructive pulmonary disease (COPD), 3) Heart problems, 4) Anxiety disorders, 5) Anemia. It's important to consult with a healthcare provider to determine the cause and receive appropriate treatment.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling dizzy quite often. What could be the reason? 😵",
        "answer": "Frequent dizziness can result from several factors, such as: 1) Inner ear issues, 2) Low blood pressure, 3) Dehydration, 4) Medication side effects, 5) Neurological conditions. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm experiencing chest pain. Should I be alarmed? 😟",
        "answer": "Chest pain can have various causes, some of which are serious. Potential causes include: 1) Heart-related issues like angina or heart attack, 2) Gastroesophageal reflux disease (GERD), 3) Muscle strain, 4) Anxiety or panic attacks, 5) Lung conditions. It's crucial to seek immediate medical attention to determine the cause and receive appropriate care.",
        "source": "medical",
        "emotional_tone": "urgent"
    },
    {
        "question": "Good afternoon, doctor. I'm having trouble sleeping lately. What can I do? 😴",
        "answer": "Difficulty sleeping can be addressed by: 1) Establishing a regular sleep schedule, 2) Creating a comfortable sleep environment, 3) Limiting caffeine and alcohol intake, 4) Avoiding screens before bedtime, 5) Practicing relaxation techniques. If sleep problems persist, consult with a healthcare provider for further evaluation.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm feeling unusually thirsty all the time. What could be causing this? 😕",
        "answer": "Excessive thirst can be a symptom of various conditions, including: 1) Diabetes mellitus, 2) Dehydration, 3) Diabetes insipidus, 4) Kidney problems, 5) Medication side effects. It's important to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Good morning, doctor. I've been experiencing frequent headaches lately. Could it be something serious? 😕",
        "answer": "Frequent headaches can have various causes, such as tension, migraines, or sinus issues. However, if they're severe, persistent, or accompanied by other symptoms like vision changes or nausea, it's important to consult a healthcare provider for a thorough evaluation.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling anxious about my upcoming surgery. Is this normal? 😟",
        "answer": "It's completely normal to feel anxious before surgery. To manage this anxiety: 1) Discuss your concerns with your surgical team, 2) Practice relaxation techniques like deep breathing or meditation, 3) Ensure you understand the procedure and recovery process, 4) Consider speaking with a counselor if anxiety persists. Remember, your feelings are valid, and support is available.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hello doctor, I'm concerned about my recent forgetfulness. Could it be something serious? 😕",
        "answer": "Experiencing forgetfulness can be unsettling. Possible causes include: 1) Stress or anxiety, 2) Sleep deprivation, 3) Nutritional deficiencies, 4) Medication side effects, 5) Underlying medical conditions like thyroid issues or, in rare cases, early dementia. It's important to consult with a healthcare provider for a thorough evaluation. You're taking a proactive step by seeking help.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm thrilled to be pregnant but also nervous about what's ahead. What should I expect? 🤰😊",
        "answer": "Congratulations on your pregnancy! It's natural to feel a mix of excitement and nervousness. During this time: 1) Schedule regular prenatal visits, 2) Maintain a balanced diet rich in essential nutrients, 3) Engage in safe, moderate exercise, 4) Educate yourself about the stages of pregnancy and childbirth, 5) Reach out to support groups or classes for expectant parents. Embrace this journey, and remember, support is available every step of the way.",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Good afternoon, doctor. I'm feeling overwhelmed by my recent diagnosis. How can I cope? 😔",
        "answer": "Receiving a new diagnosis can be overwhelming. To cope: 1) Gather information about your condition to understand it better, 2) Connect with support groups or others facing similar challenges, 3) Consider speaking with a mental health professional, 4) Practice self-care activities that bring you comfort, 5) Lean on friends and family for support. Remember, it's okay to seek help, and you're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hi doctor, I'm worried about my recent weight loss. What could be causing it? 😟",
        "answer": "Unintentional weight loss can be concerning. Potential causes include: 1) Thyroid disorders, 2) Diabetes, 3) Gastrointestinal issues, 4) Cancer, 5) Chronic infections. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm feeling constantly fatigued. What might be the reason? 😴",
        "answer": "Persistent fatigue can result from various factors, such as: 1) Anemia, 2) Sleep disorders like sleep apnea, 3) Chronic fatigue syndrome, 4) Depression or anxiety, 5) Nutritional deficiencies. A comprehensive evaluation by a healthcare provider can help identify the cause and guide appropriate treatment.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Good morning, doctor. I'm experiencing shortness of breath lately. Should I be worried? 😰",
        "answer": "Shortness of breath can be caused by various conditions, including: 1) Asthma, 2) Chronic obstructive pulmonary disease (COPD), 3) Heart problems, 4) Anxiety disorders, 5) Anemia. It's important to consult with a healthcare provider to determine the cause and receive appropriate treatment.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling dizzy quite often. What could be the reason? 😵",
        "answer": "Frequent dizziness can result from several factors, such as: 1) Inner ear issues, 2) Low blood pressure, 3) Dehydration, 4) Medication side effects, 5) Neurological conditions. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm experiencing chest pain. Should I be alarmed? 😟",
        "answer": "Chest pain can have various causes, some of which are serious. Potential causes include: 1) Heart-related issues like angina or heart attack, 2) Gastroesophageal reflux disease (GERD), 3) Muscle strain, 4) Anxiety or panic attacks, 5) Lung conditions. It's crucial to seek immediate medical attention to determine the cause and receive appropriate care.",
        "source": "medical",
        "emotional_tone": "urgent"
    },
    {
        "question": "Good afternoon, doctor. I'm having trouble sleeping lately. What can I do? 😴",
        "answer": "Difficulty sleeping can be addressed by: 1) Establishing a regular sleep schedule, 2) Creating a comfortable sleep environment, 3) Limiting caffeine and alcohol intake, 4) Avoiding screens before bedtime, 5) Practicing relaxation techniques. If sleep problems persist, consult with a healthcare provider for further evaluation.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm feeling unusually thirsty all the time. What could be causing this? 😕",
        "answer": "Excessive thirst can be a symptom of various conditions, including: 1) Diabetes mellitus, 2) Dehydration, 3) Diabetes insipidus, 4) Kidney problems, 5) Medication side effects. It's important to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
     {
        "question": "Good afternoon, doctor. I'm feeling overwhelmed by my recent diagnosis. How can I cope? 😔",
        "answer": "Receiving a new diagnosis can be overwhelming. To cope: 1) Gather information about your condition to understand it better, 2) Connect with support groups or others facing similar challenges, 3) Consider speaking with a mental health professional, 4) Practice self-care activities that bring you comfort, 5) Lean on friends and family for support. Remember, it's okay to seek help, and you're not alone in this.",
        "source": "medical",
        "emotional_tone": "compassionate"
    },
    {
        "question": "Hi doctor, I'm worried about my recent weight loss. What could be causing it? 😟",
        "answer": "Unintentional weight loss can be concerning. Potential causes include: 1) Thyroid disorders, 2) Diabetes, 3) Gastrointestinal issues, 4) Cancer, 5) Chronic infections. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm feeling constantly fatigued. What might be the reason? 😴",
        "answer": "Persistent fatigue can result from various factors, such as: 1) Anemia, 2) Sleep disorders like sleep apnea, 3) Chronic fatigue syndrome, 4) Depression or anxiety, 5) Nutritional deficiencies. A comprehensive evaluation by a healthcare provider can help identify the cause and guide appropriate treatment.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Good morning, doctor. I'm experiencing shortness of breath lately. Should I be worried? 😰",
        "answer": "Shortness of breath can be caused by various conditions, including: 1) Asthma, 2) Chronic obstructive pulmonary disease (COPD), 3) Heart problems, 4) Anxiety disorders, 5) Anemia. It's important to consult with a healthcare provider to determine the cause and receive appropriate treatment.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, I'm feeling dizzy quite often. What could be the reason? 😵",
        "answer": "Frequent dizziness can result from several factors, such as: 1) Inner ear issues, 2) Low blood pressure, 3) Dehydration, 4) Medication side effects, 5) Neurological conditions. It's essential to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm experiencing chest pain. Should I be alarmed? 😟",
        "answer": "Chest pain can have various causes, some of which are serious. Potential causes include: 1) Heart-related issues like angina or heart attack, 2) Gastroesophageal reflux disease (GERD), 3) Muscle strain, 4) Anxiety or panic attacks, 5) Lung conditions. It's crucial to seek immediate medical attention to determine the cause and receive appropriate care.",
        "source": "medical",
        "emotional_tone": "urgent"
    },
    {
        "question": "Good afternoon, doctor. I'm having trouble sleeping lately. What can I do? 😴",
        "answer": "Difficulty sleeping can be addressed by: 1) Establishing a regular sleep schedule, 2) Creating a comfortable sleep environment, 3) Limiting caffeine and alcohol intake, 4) Avoiding screens before bedtime, 5) Practicing relaxation techniques. If sleep problems persist, consult with a healthcare provider for further evaluation.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm feeling unusually thirsty all the time. What could be causing this? 😕",
        "answer": "Excessive thirst can be a symptom of various conditions, including: 1) Diabetes mellitus, 2) Dehydration, 3) Diabetes insipidus, 4) Kidney problems, 5) Medication side effects. It's important to consult with a healthcare provider for a thorough evaluation to determine the underlying cause.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hello doctor, I'm experiencing frequent urination. Should I be concerned? 😟",
        "answer": "Frequent urination can result from several factors such as urinary tract infections, diabetes, prostate issues, or even excessive fluid intake. If it’s persistent or accompanied by other symptoms like pain, fever, or blood in urine, consult a doctor for evaluation.",
        "source": "medical",
        "emotional_tone": "concerned"
    },
    {
        "question": "Hi doctor, my child has a persistent fever. When should I take them to the hospital? 🤒",
        "answer": "A persistent fever in a child can be concerning. Seek medical attention if: 1) The fever is above 102°F (39°C) and doesn't reduce with medication, 2) Your child is unusually lethargic, 3) They have difficulty breathing, 4) There are signs of dehydration, 5) The fever lasts more than three days. Trust your instincts—if you're worried, it’s best to see a doctor.",
        "source": "medical",
        "emotional_tone": "reassuring"
    },
    {
        "question": "Hello doctor, I'm considering getting a flu shot. Is it really necessary? 🤔",
        "answer": "Yes, the flu shot is highly recommended, especially for individuals at higher risk such as the elderly, pregnant women, and those with chronic illnesses. It helps prevent severe complications and reduces the spread of the virus. Protecting yourself also helps protect those around you!",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Good morning, doctor. My skin has been unusually dry and itchy. What could be the cause? 😕",
        "answer": "Dry and itchy skin can result from various factors like weather changes, dehydration, eczema, or allergic reactions. Try using a good moisturizer, drinking plenty of water, and avoiding harsh soaps. If the issue persists or worsens, a dermatologist can help identify underlying causes.",
        "source": "medical",
        "emotional_tone": "supportive"
    },
    {
        "question": "Hi doctor, I'm struggling to lose weight despite exercising regularly. What should I do? 😩",
        "answer": "Weight loss can be affected by several factors like diet, metabolism, hormones, and stress levels. Ensure you're eating a balanced diet, managing stress, getting enough sleep, and considering a medical check-up to rule out underlying conditions like thyroid issues. Keep going—you’re making progress!",
        "source": "medical",
        "emotional_tone": "encouraging"
    },
    {
        "question": "Hello doctor, I feel heart palpitations sometimes. Should I be worried? 😰",
        "answer": "Heart palpitations can be caused by stress, caffeine, dehydration, or even underlying heart conditions. If they’re frequent, last long, or come with symptoms like dizziness, chest pain, or shortness of breath, consult a doctor immediately. Stay calm, but seek an evaluation just to be safe.",
        "source": "medical",
        "emotional_tone": "concerned"
    }
]
    
    def _clean_data(self, df):
        """Clean and format the dataset"""
        # Add more sophisticated cleaning
        df = df.copy()
        
        # Add medical terminology validation
        medical_terms = set([
    # Common Conditions
    'diabetes', 'hypertension', 'arthritis', 'asthma', 'depression', 'anxiety',
    'hypothyroidism', 'hyperthyroidism', 'osteoporosis', 'fibromyalgia',
    
    # Cardiovascular
    'arrhythmia', 'tachycardia', 'bradycardia', 'atherosclerosis', 'angina',
    'myocardial infarction', 'stroke', 'thrombosis', 'embolism', 'aneurysm',
    
    # Respiratory
    'pneumonia', 'bronchitis', 'emphysema', 'COPD', 'tuberculosis',
    'sleep apnea', 'pulmonary embolism', 'pleurisy', 'bronchiectasis',
    
    # Gastrointestinal
    'gastritis', 'ulcer', 'colitis', "Crohn's disease", 'celiac disease', 'irritable bowel syndrome',
    'diverticulitis', 'hepatitis', 'cirrhosis', 'pancreatitis',
    
    # Neurological
    'migraine', 'epilepsy', "Alzheimer's disease", "Parkinson's disease", 'multiple sclerosis',
    'neuropathy', 'meningitis', 'encephalitis', 'dementia', 'concussion',
    
    # Endocrine
    'diabetes mellitus', 'hypoglycemia', "Cushing's syndrome", "Addison's disease",
    "Hashimoto's thyroiditis", "Graves' disease", 'goiter', 'pituitary disorders', 'acromegaly',
    
    # Musculoskeletal
    'osteoarthritis', 'rheumatoid arthritis', 'gout', 'systemic lupus erythematosus', 'scoliosis',
    'tendinitis', 'bursitis', 'carpal tunnel syndrome', 'sciatica', 'herniated disc',
    
    # Mental Health
    'schizophrenia', 'bipolar disorder', 'post-traumatic stress disorder', 'obsessive-compulsive disorder', 'attention deficit hyperactivity disorder',
    'eating disorders', 'anorexia nervosa', 'bulimia nervosa', 'autism spectrum disorder', 'depression',
    
    # Infectious Diseases
    'influenza', 'COVID-19', 'HIV', 'AIDS', 'malaria',
    'sepsis', 'MRSA', 'mononucleosis', 'Lyme disease', 'shingles',
    
    # Cancer Types
    'leukemia', 'lymphoma', 'melanoma', 'carcinoma', 'sarcoma',
    'glioma', 'myeloma', 'metastasis', 'neoplasm', 'tumor',
    
    # Symptoms
    'dyspnea', 'tachypnea', 'syncope', 'vertigo', 'dysphagia',
    'hemoptysis', 'hematuria', 'edema', 'cyanosis', 'jaundice',
    
    # Procedures
    'biopsy', 'endoscopy', 'colonoscopy', 'laparoscopy', 'thoracotomy',
    'angioplasty', 'catheterization', 'dialysis', 'chemotherapy', 'radiotherapy',
    
    # Medications
    'antibiotics', 'analgesics', 'antidepressants', 'antipsychotics', 'anticoagulants',
    'antihypertensives', 'bronchodilators', 'corticosteroids', 'insulin', 'statins',
    
    # Medical Specialties
    'cardiology', 'neurology', 'oncology', 'pediatrics', 'psychiatry',
    'endocrinology', 'gastroenterology', 'rheumatology', 'nephrology', 'pulmonology',
    
    # Diagnostic Tests
    'MRI', 'CT scan', 'ultrasound', 'X-ray', 'ECG',
    'EEG', 'blood test', 'urinalysis', 'biopsy', 'endoscopy'
])

        
        # Add response structure validation
        df['has_medical_terms'] = df['answer'].apply(lambda x: any(term in x.lower() for term in medical_terms))
        df['has_numbered_points'] = df['answer'].str.contains(r'\d\)')
        
        # Filter for high-quality responses
        df = df[df['has_medical_terms'] & df['has_numbered_points']].copy()
        
        # Format text with more context
        df.loc[:, 'text'] = df.apply(
            lambda x: f"### Instruction: Respond to this medical question professionally and empathetically:\n{x['question']}\n\n### Response: {x['answer']}\n\nTone: {x['emotional_tone']}", 
            axis=1
        )
        
        return df
    
    def prepare_training_data(self, df):
        """Convert dataframe to tokenized tensorflow dataset"""
        texts = df['text'].tolist()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        
        # Create input_ids and labels
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Create tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids  # For causal language modeling
        })
        
        return dataset
    
    def save_dataset(self, df, output_file):
        """Save processed dataset to disk"""
        df.to_json(output_file, orient='records', lines=True)
        print(f"Dataset saved to {output_file}")
        print(f"Total examples: {len(df)}")
        print("\nSample entries:")
        print(df['text'].head())
