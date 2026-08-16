"""
utils/medical_synonyms.py
─────────────────────────
Multilingual symptom synonym maps.

These maps convert English, Hinglish, Marathi Devanagari, and Romanized
Marathi symptom phrases into canonical English symptom terms used by the
existing TF-IDF classifier.
"""

ENGLISH_SYMPTOM_MAP = {
    "fever": "fever",
    "high fever": "high fever",
    "headache": "headache",
    "severe headache": "severe headache",
    "cough": "cough",
    "cold": "cold",
    "runny nose": "runny nose",
    "nasal congestion": "nasal congestion",
    "seasonal nasal congestion": "nasal congestion",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "vomiting": "vomiting",
    "nausea": "nausea",
    "dizziness": "dizziness",
    "shortness of breath": "shortness of breath",
    "difficulty breathing": "difficulty breathing",
    "temporary inability to inhale": "temporary inability to breathe",
    "temporary inability to breathe": "temporary inability to breathe",
    "abdominal pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "back pain": "back pain",
    "lower back pain": "lower back pain",
    "low back pain": "lower back pain",
    "flank pain": "flank pain",
    "chest pain": "chest pain",
    "sore throat": "sore throat",
    "burning urination": "burning urination",
    "difficulty urinating": "difficulty urinating",
    "trouble urinating": "difficulty urinating",
    "slow urine": "difficulty urinating",
    "slow urine stream": "difficulty urinating",
    "slow urination": "difficulty urinating",
    "urine takes time": "difficulty urinating",
    "longer time for urine": "difficulty urinating",
    "taking long time to urinate": "difficulty urinating",
    "urinary retention": "difficulty urinating",
    "cannot urinate": "difficulty urinating",
    "unable to urinate": "difficulty urinating",
    "frequent urination": "frequent urination",
    "blood in urine": "blood in urine",
    "diarrhea": "diarrhea",
    "loose motion": "diarrhea",
    "fatigue": "fatigue",
    "weakness": "weakness",
    "itching": "itching",
    "jaundice": "jaundice",
    "yellow eyes": "jaundice",
    "yellow skin": "jaundice",
    "dark urine": "dark urine",
    "body pain": "body pain",
    "body ache": "body pain",
    "joint pain": "joint pain",
    "wheezing": "wheezing",
    "chest tightness": "chest tightness",
    "heartburn": "heartburn",
    "chest burning": "heartburn",
    "drooping eyelid": "drooping eyelid",
    "eyelid drooping": "drooping eyelid",
}


HINGLISH_SYMPTOM_MAP = {
    "bukhar": "fever",
    "taap": "fever",
    "sar dard": "headache",
    "sir dard": "headache",
    "khansi": "cough",
    "sardi": "cold",
    "zukam": "cold",
    "zukaam": "cold",
    "ulti": "vomiting",
    "matli": "nausea",
    "jee michalna": "nausea",
    "ji michalna": "nausea",
    "jee machalna": "nausea",
    "seene mein jalan": "heartburn",
    "seene me jalan": "heartburn",
    "chakkar": "dizziness",
    "saans phoolna": "shortness of breath",
    "saans lene me dikkat": "difficulty breathing",
    "saans lene mein dikkat": "difficulty breathing",
    "saans lene me takleef": "difficulty breathing",
    "pet dard": "abdominal pain",
    "pait dard": "abdominal pain",
    "kamar dard": "lower back pain",
    "pith dard": "back pain",
    "chhati dard": "chest pain",
    "chati dard": "chest pain",
    "gala dard": "sore throat",
    "gala kharab": "sore throat",
    "gaka kharab": "sore throat",
    "peshab me jalan": "burning urination",
    "peshab mein jalan": "burning urination",
    "peshab me dikkat": "difficulty urinating",
    "peshab mein dikkat": "difficulty urinating",
    "peshab ruk ruk": "difficulty urinating",
    "peshab ruk ruk ke": "difficulty urinating",
    "peshab dheere": "difficulty urinating",
    "peshab me khoon": "blood in urine",
    "bar bar peshab": "frequent urination",
    "dast": "diarrhea",
    "thakan": "fatigue",
    "kamzori": "weakness",
    "khujli": "itching",
    "peeli aankh": "jaundice",
    "peeli skin": "jaundice",
    "piliya": "jaundice",
    "gadha peshab": "dark urine",
    "dark peshab": "dark urine",
    "body pain": "body pain",
    "badan dard": "body pain",
}


MARATHI_SYMPTOM_MAP = {
    "ताप": "fever",
    "जास्त ताप": "high fever",
    "डोकेदुखी": "headache",
    "तीव्र डोकेदुखी": "severe headache",
    "खोकला": "cough",
    "सर्दी": "cold",
    "उलटी": "vomiting",
    "मळमळ": "nausea",
    "चक्कर": "dizziness",
    "पोटदुखी": "abdominal pain",
    "पोट दुखणे": "abdominal pain",
    "कंबरदुखी": "lower back pain",
    "पाठदुखी": "back pain",
    "छातीत दुखणे": "chest pain",
    "छाती दुखणे": "chest pain",
    "घसा दुखणे": "sore throat",
    "श्वास घेण्यास त्रास": "difficulty breathing",
    "श्वास घेण्यास अडचण": "difficulty breathing",
    "धाप लागणे": "shortness of breath",
    "जुलाब": "diarrhea",
    "लघवीला जळजळ": "burning urination",
    "लघवीला त्रास": "difficulty urinating",
    "लघवी होत नाही": "difficulty urinating",
    "लघवीला वेळ लागतो": "difficulty urinating",
    "लघवीत रक्त": "blood in urine",
    "वारंवार लघवी": "frequent urination",
    "थकवा": "fatigue",
    "अशक्तपणा": "weakness",
    "खाज": "itching",
    "कावीळ": "jaundice",
    "पिवळे डोळे": "jaundice",
    "गडद लघवी": "dark urine",
    "अंगदुखी": "body pain",
    "सांधेदुखी": "joint pain",
}


ROMAN_MARATHI_SYMPTOM_MAP = {
    "taap": "fever",
    "jasta taap": "high fever",
    "dokedukhi": "headache",
    "dok dukhi": "headache",
    "tivra dokedukhi": "severe headache",
    "khokla": "cough",
    "sardi": "cold",
    "ulti": "vomiting",
    "malmal": "nausea",
    "chakkar": "dizziness",
    "potdukhi": "abdominal pain",
    "pot dukhane": "abdominal pain",
    "kambardukhi": "lower back pain",
    "kambar dukhane": "lower back pain",
    "pathdukhi": "back pain",
    "chatit dukhane": "chest pain",
    "chati dukhane": "chest pain",
    "ghasa dukhane": "sore throat",
    "shwas ghenyas tras": "difficulty breathing",
    "shwas ghyayla tras": "difficulty breathing",
    "dhap lagne": "shortness of breath",
    "julab": "diarrhea",
    "laghvila jaljal": "burning urination",
    "laghvila tras": "difficulty urinating",
    "laghvi hot nahi": "difficulty urinating",
    "laghvila vel lagto": "difficulty urinating",
    "laghvit rakt": "blood in urine",
    "waranwar laghvi": "frequent urination",
    "thakva": "fatigue",
    "ashaktpana": "weakness",
    "khaj": "itching",
    "kavil": "jaundice",
    "pivle dole": "jaundice",
    "gadad laghvi": "dark urine",
    "angdukhi": "body pain",
    "sandhedukhi": "joint pain",
}


# Narrow, validated misspellings handled by the deterministic normalizer.
# These are intentionally kept separate from the fuzzy vocabulary so they
# cannot broaden fuzzy candidate generation or lower its safety threshold.
DETERMINISTIC_TYPO_MAP = {
    "coughh": "cough",
    "fevr": "fever",
    "headeche": "headache",
    "nausous": "nausea",
    "wheezng": "wheezing",
}


FILLER_WORDS = {
    "i", "have", "has", "am", "is", "are", "and", "or", "with", "the", "a",
    "an", "my", "me", "mujhe", "hai", "aur", "ko", "mein", "me", "ka",
    "ki", "ke", "mala", "aahe", "ani", "ahe", "hoto", "hote", "hot",
    "मला", "आहे", "आणि", "होतो", "होते",
}


def all_symptom_maps() -> dict[str, dict[str, str]]:
    """Return all synonym maps grouped by language family."""
    return {
        "english": ENGLISH_SYMPTOM_MAP,
        "hinglish": HINGLISH_SYMPTOM_MAP,
        "marathi_devanagari": MARATHI_SYMPTOM_MAP,
        "romanized_marathi": ROMAN_MARATHI_SYMPTOM_MAP,
    }


def merged_symptom_map() -> dict[str, str]:
    """Return one merged phrase-to-canonical symptom map."""
    merged: dict[str, str] = {}
    for symptom_map in all_symptom_maps().values():
        merged.update(symptom_map)
    merged.update(DETERMINISTIC_TYPO_MAP)
    return merged


def canonical_symptoms() -> set[str]:
    """Return the canonical English symptom vocabulary."""
    return set(merged_symptom_map().values())
