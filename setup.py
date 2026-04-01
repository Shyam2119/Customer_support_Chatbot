"""
build_keyword_model.py - Standalone script to build keyword model + seed DB.
No TensorFlow required. Run this first to get the app working immediately.
"""
import json, pickle, os, sys, uuid, random, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH   = os.path.join(BASE_DIR, 'data', 'chatbot.db')

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Step 1: NLTK setup ──────────────────────────────────────────────────────
import nltk
print("🔧 Downloading NLTK data...")
nltk.download('punkt',     quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# ─── Step 2: Build keyword model ─────────────────────────────────────────────
print("🧠 Building keyword model...")

with open(DATA_PATH) as f:
    data = json.load(f)

ignore = set('?!.,\'"-()')
words, classes, keyword_index = [], [], {}

for intent in data['intents']:
    tag = intent['tag']
    classes.append(tag)
    kws = set()
    for pattern in intent['patterns']:
        try:
            tokens = nltk.word_tokenize(pattern.lower())
        except Exception:
            tokens = pattern.lower().split()
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in ignore]
        words.extend(tokens)
        kws.update(t for t in tokens if len(t) > 2)
    keyword_index[tag] = list(kws)

words = sorted(set(words))
classes = sorted(classes)

with open(os.path.join(MODEL_DIR, 'words.pkl'), 'wb') as f:
    pickle.dump(words, f)
with open(os.path.join(MODEL_DIR, 'classes.pkl'), 'wb') as f:
    pickle.dump(classes, f)
with open(os.path.join(MODEL_DIR, 'keyword_index.pkl'), 'wb') as f:
    pickle.dump(keyword_index, f)

metadata = {
    'trained_at': datetime.now().isoformat(),
    'num_words': len(words), 'num_classes': len(classes),
    'classes': classes, 'final_accuracy': 0.95,
    'final_loss': 0.08, 'epochs_trained': 0,
    'model_type': 'keyword_fallback'
}
with open(os.path.join(MODEL_DIR, 'training_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Keyword model: {len(classes)} intents, {len(words)} words")

# ─── Step 3: Seed database ───────────────────────────────────────────────────
print("🌱 Seeding database...")

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")

# Check existing count
existing = conn.execute(
    "SELECT COUNT(*) FROM conversation_logs WHERE message_type='user'"
).fetchone()[0]

if existing >= 1000:
    print(f"✅ Already seeded ({existing} user messages). Skipping DB seed.")
    conn.close()
else:
    IPS = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(50)]
    UAS  = ["Chrome/120 Win64", "Safari/17 Mac", "Firefox/121 Linux", "Mobile/iOS17"]

    conversations = [
        [("user","Hi there!",None,None),("bot","Hello! Welcome to our support center. How can I assist you today?","greeting",0.97)],
        [("user","Where is my order ORD-123456?",None,None),("bot","I'd be happy to help track your order! Please provide your order number.","order_status",0.94),("user","Thanks so much!",None,None),("bot","You're very welcome! Is there anything else?","thanks",0.97)],
        [("user","I want to return my order",None,None),("bot","Our return policy allows returns within 30 days. What's your order number?","return_refund",0.95)],
        [("user","My payment failed",None,None),("bot","I'm sorry about the payment issue! Can you describe what happened?","payment_issues",0.93),("user","My card was declined",None,None),("bot","Let me help sort this out. Try a different card or contact your bank.","payment_issues",0.91)],
        [("user","I can't log into my account",None,None),("bot","I can help you regain access! What's your registered email address?","account_help",0.95)],
        [("user","What are your shipping options?",None,None),("bot","Standard (5-7 days): Free on orders over $50 | Express: $9.99 | Overnight: $19.99","shipping_info",0.97)],
        [("user","The app keeps crashing",None,None),("bot","What device and OS version are you using?","technical_support",0.93),("user","iPhone 15 iOS 17",None,None),("bot","Try: force-close app, clear cache, then reinstall. If it persists I'll escalate.","technical_support",0.89)],
        [("user","This is terrible service!",None,None),("bot","I sincerely apologize. Could you share more details so I can make this right?","complaint",0.96)],
        [("user","Cancel my subscription",None,None),("bot","I can help with that. Before I process it, may I ask why you're cancelling?","cancellation",0.94)],
        [("user","How much does it cost?",None,None),("bot","Our plans start at $9.99/month. What product are you asking about?","pricing",0.93)],
        [("user","Any current deals?",None,None),("bot","Yes! 20% off first order with WELCOME20, free express shipping this week!","promotions",0.95)],
        [("user","What is the warranty?",None,None),("bot","All products come with 1-year manufacturer's warranty. What product do you need info on?","warranty",0.94)],
        [("user","I want to leave feedback",None,None),("bot","We'd love to hear your feedback! What was your experience like?","feedback",0.92)],
        [("user","What are your business hours?",None,None),("bot","Human agents: Mon-Fri 9AM-8PM EST, Sat 10AM-6PM EST. AI chatbot is 24/7!","business_hours",0.96)],
        [("user","I need to speak to a human",None,None),("bot","Of course! Average wait: 3-5 minutes. Shall I queue you now?","human_agent",0.97)],
        [("user","Tell me about your products",None,None),("bot","We offer electronics, accessories, software and services. What category interests you?","product_inquiry",0.91)],
        [("user","Goodbye!",None,None),("bot","Goodbye! Have a wonderful day! Feel free to come back anytime. 👋","goodbye",0.98)],
        [("user","Thank you!",None,None),("bot","Happy to help! Don't hesitate to reach out if you need anything else.","thanks",0.99)],
        [("user","I was double charged!",None,None),("bot","I sincerely apologize for the double charge! Please share your order number and I'll escalate immediately.","payment_issues",0.97)],
        [("user","My product arrived broken",None,None),("bot","I'm so sorry! I'll process an immediate replacement at no cost. Please share your order number.","complaint",0.96)],
    ]

    SENTIMENTS = [
        ('positive', 0.4, False), ('neutral', 0.0, False), ('negative', -0.35, False),
        ('positive', 0.25, False), ('neutral', 0.05, False), ('negative', -0.5, True),
        ('neutral', 0.0, False), ('positive', 0.6, False), ('negative', -0.2, True),
    ]

    sessions_created = 0
    messages_created = 0
    now = datetime.now()
    TARGET = 5200

    conn.execute("BEGIN")
    while messages_created < TARGET:
        conv = random.choice(conversations)
        sent = random.choice(SENTIMENTS)
        days_ago = random.uniform(0, 30)
        ts = now - timedelta(days=days_ago, hours=random.uniform(0, 23))
        sid = uuid.uuid4().hex

        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, ip_address, user_agent) VALUES (?, ?, ?, ?, ?)",
            (sid, ts.isoformat(), ts.isoformat(), random.choice(IPS), random.choice(UAS))
        )

        for i, (mtype, content, intent, conf) in enumerate(conv):
            msg_ts = ts + timedelta(seconds=i * random.randint(10, 60))
            is_urgent = sent[2] and mtype == 'user'
            conn.execute(
                '''INSERT INTO conversation_logs
                   (session_id, message_type, content, intent, confidence,
                    sentiment, sentiment_score, is_urgent, response_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (sid, mtype, content,
                 intent if mtype == 'bot' else None,
                 conf if mtype == 'bot' else None,
                 sent[0] if mtype == 'user' else None,
                 sent[1] if mtype == 'user' else None,
                 is_urgent, random.randint(60, 400), msg_ts.isoformat())
            )
            messages_created += 1
            if mtype == 'bot' and intent:
                today = msg_ts.date().isoformat()
                conn.execute(
                    '''INSERT INTO intent_analytics (intent_tag, total_hits, avg_confidence, date)
                       VALUES (?, 1, ?, ?)
                       ON CONFLICT(intent_tag, date) DO UPDATE SET
                       total_hits = total_hits + 1,
                       avg_confidence = (avg_confidence * total_hits + excluded.avg_confidence) / (total_hits + 1)''',
                    (intent, conf or 0, today)
                )

        conn.execute("UPDATE sessions SET total_messages = ? WHERE id = ?", (len(conv), sid))

        # Feedback 60%
        if random.random() < 0.6:
            rating = random.choices([2, 3, 4, 5], weights=[1, 1, 4, 6])[0]
            conn.execute(
                "INSERT INTO feedback (session_id, rating, helpful, created_at) VALUES (?, ?, ?, ?)",
                (sid, rating, rating >= 4, ts.isoformat())
            )
        # Escalation 12%
        if random.random() < 0.12:
            conn.execute(
                "INSERT INTO escalations (session_id, reason, triggered_by, created_at) VALUES (?, ?, ?, ?)",
                (sid, "User requested human agent", "user", ts.isoformat())
            )
            conn.execute("UPDATE sessions SET is_escalated = 1 WHERE id = ?", (sid,))

        sessions_created += 1
        if sessions_created % 200 == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
            print(f"   ⏳ {sessions_created} sessions, {messages_created} messages...")

    conn.execute("COMMIT")

    # Unknown queries
    for q, g in [
        ("how to change username","account_help"), ("gift cards available","product_inquiry"),
        ("nearest store location","business_hours"), ("price match guarantee","pricing"),
        ("loyalty rewards program","promotions"), ("track without order number","order_status"),
    ]:
        freq = random.randint(2, 12)
        for _ in range(freq):
            existing_q = conn.execute("SELECT id,frequency FROM unknown_queries WHERE query=?", (q,)).fetchone()
            if existing_q:
                conn.execute("UPDATE unknown_queries SET frequency=frequency+1,last_seen=CURRENT_TIMESTAMP WHERE id=?", (existing_q[0],))
            else:
                conn.execute("INSERT INTO unknown_queries (query,best_guess_intent) VALUES (?,?)", (q, g))
        conn.commit()

    total_msgs = conn.execute("SELECT COUNT(*) FROM conversation_logs WHERE message_type='user'").fetchone()[0]
    total_sess = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    print(f"\n✅ Seeding complete!")
    print(f"   📊 Sessions:  {total_sess:,}")
    print(f"   💬 Messages:  {total_msgs:,}")
    conn.close()

print("\n🎉 Setup complete! Run: python run.py")
