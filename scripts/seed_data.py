"""
seed_data.py - Seed the SQLite database with 5,000+ realistic conversation logs
for demonstration and analytics testing.

Run:   python scripts/seed_data.py
Re-run safe: skips if already has >= 1000 user messages.
"""

import sys
import os
import random
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db

# ─── Conversation Templates ───────────────────────────────────────────────────
# Each template is a list of (msg_type, content, intent, confidence) tuples.
CONV_TEMPLATES = [
    # ── Greetings / Hello
    [("user","Hi there!","greeting",0.98),("bot","Hello! Welcome to our support center. How can I assist you today?","greeting",0.98)],
    [("user","Hey, good morning","greeting",0.96),("bot","Good morning! I'm your virtual assistant. What can I help you with?","greeting",0.96)],
    [("user","Hello","greeting",0.99),("bot","Hi there! I'm here to help. What do you need assistance with?","greeting",0.99)],

    # ── Order Status
    [("user","Where is my order?","order_status",0.95),("bot","I'd be happy to help track your order! Please provide your order number.","order_status",0.95),
     ("user","My order is ORD-987654","order_status",0.93),("bot","Looking up order #ORD-987654... Standard delivery: estimated 3-5 days. You'll receive a tracking email shortly.","order_status",0.93),
     ("user","Thanks!","thanks",0.97),("bot","You're very welcome! Is there anything else I can help you with?","thanks",0.97)],

    [("user","Track my package please","order_status",0.92),("bot","To check your order status, I'll need your order number. Can you provide that?","order_status",0.92)],
    [("user","When will my order arrive?","order_status",0.91),("bot","Let me help you track that! Please share your order ID or tracking number.","order_status",0.91)],
    [("user","My order is late","order_status",0.88),("bot","I'm sorry to hear that! Delays can happen. Please share your order number and I'll look into it right away.","order_status",0.88)],
    [("user","Order not received","order_status",0.86),("bot","I apologize for the inconvenience! Please provide your order number so I can investigate this for you.","order_status",0.86)],

    # ── Return / Refund
    [("user","I want to return my order","return_refund",0.96),("bot","Our return policy allows returns within 30 days. I'll need your order number. What's that?","return_refund",0.96),
     ("user","Order number 4567890","return_refund",0.89),("bot","Thank you! I've initiated a return request for order #4567890. You'll receive a prepaid return label by email within 24 hours. Refund processed in 5-7 business days.","return_refund",0.89)],

    [("user","How do I get a refund?","return_refund",0.94),("bot","For a refund, I'll need your order number. Refunds are typically processed within 5-7 business days.","return_refund",0.94)],
    [("user","I want to exchange this product","return_refund",0.88),("bot","I can help with an exchange! Please provide your order number and let me know what you'd like to exchange to.","return_refund",0.88)],
    [("user","Return policy?","return_refund",0.91),("bot","Our return policy allows returns within 30 days of purchase for most items. Items must be in original condition.","return_refund",0.91)],
    [("user","Cancel and refund my order","return_refund",0.93),("bot","I can process a cancellation and refund. Please share your order number and I'll take care of it immediately.","return_refund",0.93)],

    # ── Payment Issues
    [("user","My payment failed","payment_issues",0.94),("bot","I'm sorry about the payment issue! Can you describe what happened when you tried to pay?","payment_issues",0.94),
     ("user","Card was declined","payment_issues",0.90),("bot","Let me help sort this out. Try a different card, or contact your bank to ensure there are no blocks on international payments.","payment_issues",0.90)],

    [("user","I was double charged","payment_issues",0.96),("bot","I sincerely apologize for the double charge! This is high priority. Please share your order number and email and I'll escalate immediately.","payment_issues",0.96)],
    [("user","Payment not working","payment_issues",0.91),("bot","Let me assist with your billing concern. Are you getting an error message? What payment method are you using?","payment_issues",0.91)],
    [("user","Wrong amount charged","payment_issues",0.89),("bot","I apologize for the billing discrepancy. Please share the order number and I'll review the charge immediately.","payment_issues",0.89)],
    [("user","How do I update my payment method?","payment_issues",0.87),("bot","You can update your payment method in Account Settings → Billing. Would you like me to guide you through the steps?","payment_issues",0.87)],

    # ── Account Help
    [("user","I can't log in","account_help",0.96),("bot","I can help you regain access! For password resets, I can send a reset link to your registered email. What's your email?","account_help",0.96),
     ("user","test@example.com","account_help",0.93),("bot","A password reset link has been sent to test@example.com. Please check your inbox (and spam folder). The link expires in 1 hour.","account_help",0.93)],

    [("user","Forgot my password","account_help",0.97),("bot","No problem! I can send a password reset link to your email. What's your registered email address?","account_help",0.97)],
    [("user","Account locked","account_help",0.93),("bot","I'll help unlock your account right away. Could you verify your registered email or the last 4 digits of your payment method?","account_help",0.93)],
    [("user","Change my email address","account_help",0.88),("bot","To change your email, go to Account Settings → Personal Information. Would you like step-by-step guidance?","account_help",0.88)],
    [("user","How do I delete my account?","account_help",0.85),("bot","Before we process an account deletion, may I ask what's prompting this decision? We may be able to resolve your concern.","account_help",0.85)],

    # ── Shipping Info
    [("user","What are your shipping options?","shipping_info",0.97),("bot","We offer: Free Standard (5-7 days) on orders over $50 | Express (2-3 days) $9.99 | Overnight $19.99. International shipping available to 50+ countries.","shipping_info",0.97)],
    [("user","Do you offer free shipping?","shipping_info",0.95),("bot","Yes! Free standard shipping on all orders over $50. Express and overnight options available at additional cost.","shipping_info",0.95)],
    [("user","How long does standard shipping take?","shipping_info",0.93),("bot","Standard shipping typically takes 5-7 business days within the US. Express is 2-3 days and overnight is next day by 6pm.","shipping_info",0.93)],
    [("user","Can I ship internationally?","shipping_info",0.91),("bot","Yes! We ship to 50+ countries. International rates vary by destination. Enter your address at checkout for an exact quote.","shipping_info",0.91)],

    # ── Technical Support
    [("user","The app keeps crashing","technical_support",0.94),("bot","I'm sorry you're experiencing technical difficulties! What device and OS version are you using?","technical_support",0.94),
     ("user","iPhone 15, iOS 17","technical_support",0.89),("bot","Thank you! Please try: 1) Force-close the app 2) Clear cache in Settings 3) Reinstall. If it persists, I'll escalate to our iOS team.","technical_support",0.89)],

    [("user","Website is down","technical_support",0.92),("bot","I apologize for the inconvenience! I'm checking our system status now. Are you seeing a specific error message?","technical_support",0.92)],
    [("user","Error 404 on your website","technical_support",0.89),("bot","A 404 error means the page wasn't found. Please share the exact URL you're trying to access so I can help.","technical_support",0.89)],
    [("user","Can't upload my profile picture","technical_support",0.85),("bot","Photo upload issues are usually browser-related. Try: clear cache, use a different browser, or ensure the image is under 5MB in JPG/PNG format.","technical_support",0.85)],
    [("user","Notifications not working","technical_support",0.82),("bot","For notification issues: 1) Check app notification permissions in phone Settings 2) Ensure Do Not Disturb is off 3) Try re-enabling in app settings.","technical_support",0.82)],

    # ── Complaints
    [("user","This is terrible service!","complaint",0.97),("bot","I sincerely apologize for the experience you've had. Your feedback is important to us. Could you share more details so I can make this right?","complaint",0.97)],
    [("user","I'm very disappointed with your product","complaint",0.94),("bot","I'm truly sorry to hear this. You deserve better. Can you tell me what happened so I can escalate this appropriately?","complaint",0.94)],
    [("user","Worst experience ever","complaint",0.93),("bot","I apologize and I understand your frustration. Let me escalate this to ensure it's handled with the urgency it deserves.","complaint",0.93)],
    [("user","Your product arrived broken","complaint",0.96),("bot","I'm so sorry your product arrived damaged! I'll process an immediate replacement at no cost. Please share your order number.","complaint",0.96)],

    # ── Cancellation
    [("user","Cancel my subscription","cancellation",0.96),("bot","I can help with that cancellation. Before I process it, may I ask what's prompting you to cancel?","cancellation",0.96)],
    [("user","I want to cancel my order","cancellation",0.94),("bot","I can process that cancellation. Could you provide your order number so I can check if it's eligible for cancellation?","cancellation",0.94)],
    [("user","Stop charging my card","cancellation",0.93),("bot","I'll stop the charges right away. To process this, could you provide your account email or order number?","cancellation",0.93)],

    # ── Pricing
    [("user","How much does it cost?","pricing",0.94),("bot","Our plans start at $9.99/month. We also have an annual plan at $99/year (saving 17%). Which product are you asking about?","pricing",0.94)],
    [("user","Do you have any discounts?","pricing",0.91),("bot","We do! Use code WELCOME20 for 20% off your first order. We also offer student discounts and bundle deals. Tell me more about what you need.","pricing",0.91)],
    [("user","What's the pricing for the premium plan?","pricing",0.93),("bot","The Premium plan is $24.99/month or $249/year. It includes unlimited storage, priority support, and advanced analytics.","pricing",0.93)],

    # ── Promotions
    [("user","Any current deals?","promotions",0.95),("bot","Yes! Current offers: 20% off first order (WELCOME20), Free express shipping this week, Buy 2 get 1 free on accessories. Sign up for our newsletter for exclusive deals!","promotions",0.95)],
    [("user","Promo code not working","promotions",0.88),("bot","I'm sorry the code isn't working! Let me verify it for you. Which code are you using and what error are you seeing?","promotions",0.88)],
    [("user","Is there a student discount?","promotions",0.86),("bot","Yes! We offer 15% student discount year-round. Verify your student status through our Student Deals page and the discount applies automatically.","promotions",0.86)],

    # ── Warranty
    [("user","What is the warranty on your products?","warranty",0.95),("bot","All our products come with a 1-year manufacturer's warranty. Extended 2 and 3-year plans are also available. What product do you need info on?","warranty",0.95)],
    [("user","My product is defective","warranty",0.93),("bot","I'm sorry to hear that! If your product is defective within the warranty period, I'll initiate a replacement. Please share your order number.","warranty",0.93)],
    [("user","How do I file a warranty claim?","warranty",0.91),("bot","To file a warranty claim: 1) Provide your order number 2) Describe the defect 3) Share a photo if possible. I'll process it right away!","warranty",0.91)],

    # ── Feedback
    [("user","I want to leave a review","feedback",0.93),("bot","We'd love to hear your feedback! You can rate your experience on our website or tell me directly — what was your experience like?","feedback",0.93),
     ("user","It was great, very helpful","feedback",0.96),("bot","Thank you so much! We're thrilled you had a great experience. Your feedback helps us improve!","feedback",0.96)],

    [("user","I want to give feedback","feedback",0.91),("bot","Thank you for wanting to share feedback! Your opinion matters greatly. Please rate your experience (1-5) and share any comments.","feedback",0.91)],

    # ── Business Hours
    [("user","What are your business hours?","business_hours",0.96),("bot","Human agents: Mon-Fri 9AM-8PM EST, Sat 10AM-6PM EST. But I'm available 24/7 to assist you anytime!","business_hours",0.96)],
    [("user","Are you open on weekends?","business_hours",0.93),("bot","Our live agents work Saturday 10AM-6PM EST. Our AI chatbot (me!) is available 24/7, 365 days a year.","business_hours",0.93)],

    # ── Human Agent
    [("user","I want to speak to a human","human_agent",0.97),("bot","Of course! I'll connect you with a live agent. Agents available Mon-Fri 9AM-8PM EST. Average wait: 3-5 minutes. Shall I queue you now?","human_agent",0.97)],
    [("user","Let me talk to your manager","human_agent",0.95),("bot","I understand. Let me escalate this to a senior agent. Could you briefly describe your issue so they're prepared?","human_agent",0.95)],
    [("user","This bot is useless, real person please","human_agent",0.94),("bot","I apologize for the frustration! Transferring you to a live agent now. Please hold...","human_agent",0.94)],

    # ── Product Inquiry
    [("user","Tell me about your products","product_inquiry",0.92),("bot","We offer a wide range of products! What category are you interested in — electronics, accessories, software, or services?","product_inquiry",0.92)],
    [("user","Is this product available?","product_inquiry",0.89),("bot","I'd be glad to check availability! Which product are you looking for?","product_inquiry",0.89)],
    [("user","What features does your software have?","product_inquiry",0.86),("bot","Our software includes: real-time analytics, AI automation, cloud sync, 24/7 support, and 50+ integrations. Want a full feature list?","product_inquiry",0.86)],

    # ── Thanks / Goodbye
    [("user","Thank you so much!","thanks",0.98),("bot","You're very welcome! Don't hesitate to reach out if you need anything else. Have a wonderful day!","thanks",0.98)],
    [("user","Bye, that was helpful","goodbye",0.96),("bot","Goodbye! We're always here if you need us. Take care! 👋","goodbye",0.96)],
    [("user","Perfect, thanks!","thanks",0.97),("bot","Happy to help! Anything else I can do for you?","thanks",0.97)],
    [("user","That's all I needed","goodbye",0.94),("bot","Great! Feel free to come back anytime. Have a great day!","goodbye",0.94)],
]

SENTIMENTS = [
    ('positive', 0.45, False),
    ('neutral',  0.05, False),
    ('negative', -0.35, False),
    ('positive', 0.25, False),
    ('neutral',  0.00, False),
    ('negative', -0.50, True),
    ('neutral',  0.10, False),
    ('positive', 0.60, False),
    ('neutral',  0.00, False),
    ('negative', -0.20, True),
]

IPS = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(100)]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14) Safari/537",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17) AppleWebKit/605",
    "Mozilla/5.0 (Android 14; Mobile) Chrome/120",
    "PostmanRuntime/7.36.0",
]

FEEDBACK_TEXTS = [
    "Very helpful, resolved my issue quickly!",
    "Great support, thank you so much.",
    "Resolved my problem instantly. Excellent!",
    "Good but could be faster.",
    "Not very helpful, had to contact again.",
    "The bot understood my issue right away.",
    "I wish I could speak to a human faster.",
    None, None, None
]


def seed_database():
    db = get_db()

    # Check existing data → skip if already seeded
    stats = db.get_dashboard_stats()
    existing = stats['overview']['total_messages']
    if existing >= 1000:
        print(f"✅ Already seeded ({existing} messages). Skipping.")
        return

    print(f"🌱 Seeding database with 5,000+ conversation log rows...")

    sessions_created = 0
    messages_created = 0
    TARGET = 5200  # generate slightly above 5k

    # Spread sessions over last 30 days
    now = datetime.now()

    import sqlite3
    from app.database import DB_PATH
    from contextlib import contextmanager

    # We'll use bulk writes for performance
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        while messages_created < TARGET:
            # Pick a random conversation template
            conv = random.choice(CONV_TEMPLATES)
            sentiment_row = random.choice(SENTIMENTS)

            # Random timestamp in the past 30 days
            days_ago = random.uniform(0, 30)
            session_ts = now - timedelta(days=days_ago, hours=random.uniform(0, 23))

            session_id = None
            try:
                cur = conn.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, ip_address, user_agent) VALUES (?, ?, ?, ?, ?)",
                    (
                        __import__('uuid').uuid4().hex,
                        session_ts.isoformat(),
                        session_ts.isoformat(),
                        random.choice(IPS),
                        random.choice(USER_AGENTS),
                    )
                )
                session_id = cur.lastrowid

                # Walk conversation
                for idx, turn in enumerate(conv):
                    msg_type, content, intent, confidence = turn
                    is_urgent = sentiment_row[2] and msg_type == 'user'
                    msg_ts = session_ts + timedelta(seconds=idx * random.randint(8, 60))

                    conn.execute(
                        '''INSERT INTO conversation_logs
                           (session_id, message_type, content, intent, confidence,
                            sentiment, sentiment_score, is_urgent, response_time_ms, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            session_id,
                            msg_type,
                            content,
                            intent if msg_type == 'bot' else None,
                            confidence if msg_type == 'bot' else None,
                            sentiment_row[0] if msg_type == 'user' else None,
                            sentiment_row[1] if msg_type == 'user' else None,
                            is_urgent,
                            random.randint(60, 420),
                            msg_ts.isoformat(),
                        )
                    )
                    messages_created += 1

                    # Update intent analytics
                    today = msg_ts.date().isoformat()
                    if msg_type == 'bot' and intent:
                        conn.execute(
                            '''INSERT INTO intent_analytics (intent_tag, total_hits, avg_confidence, date)
                               VALUES (?, 1, ?, ?)
                               ON CONFLICT(intent_tag, date) DO UPDATE SET
                                 total_hits = total_hits + 1,
                                 avg_confidence = (avg_confidence * total_hits + excluded.avg_confidence) / (total_hits + 1)''',
                            (intent, confidence, today)
                        )

                # Update session total
                conn.execute(
                    "UPDATE sessions SET total_messages = ? WHERE id = ?",
                    (len(conv), session_id)
                )

                # Feedback (~60% of sessions)
                if random.random() < 0.6:
                    rating = random.choices([2, 3, 4, 5], weights=[1, 1, 4, 6])[0]
                    feedback_ts = session_ts + timedelta(minutes=random.randint(2, 30))
                    conn.execute(
                        '''INSERT INTO feedback (session_id, rating, feedback_text, helpful, created_at)
                           VALUES (?, ?, ?, ?, ?)''',
                        (
                            session_id,
                            rating,
                            random.choice(FEEDBACK_TEXTS),
                            rating >= 4,
                            feedback_ts.isoformat(),
                        )
                    )

                # Escalations (~12% of sessions)
                if random.random() < 0.12:
                    reason = random.choice([
                        "User requested human agent",
                        "Complex issue beyond bot scope",
                        "User expressed high frustration",
                        "Multiple unresolved queries"
                    ])
                    conn.execute(
                        "INSERT INTO escalations (session_id, reason, triggered_by, created_at) VALUES (?, ?, ?, ?)",
                        (session_id, reason, random.choice(['user', 'system']), session_ts.isoformat())
                    )
                    conn.execute("UPDATE sessions SET is_escalated = 1 WHERE id = ?", (session_id,))

                sessions_created += 1
                if sessions_created % 100 == 0:
                    conn.commit()
                    print(f"   ⏳ {sessions_created} sessions, {messages_created} messages...")

            except Exception as e:
                print(f"   ⚠️  Error in session: {e}")
                continue

        # Final commit
        conn.commit()

    finally:
        conn.close()

    # Add unknown queries
    print("📝 Adding unknown query samples...")
    unknown_queries = [
        ("how do i change my username", "account_help"),
        ("what countries do you ship to", "shipping_info"),
        ("can i get a price match", "pricing"),
        ("do you have gift cards", "product_inquiry"),
        ("where is my nearest store", "business_hours"),
        ("can i track without order number", "order_status"),
        ("what happens after i return", "return_refund"),
        ("is there a loyalty program", "promotions"),
        ("can i change my delivery address", "order_status"),
        ("how long is the warranty on batteries", "warranty"),
        ("do you have live chat", "human_agent"),
        ("why was my card flagged", "payment_issues"),
    ]
    db2 = get_db()
    for query, guess in unknown_queries:
        freq = random.randint(1, 15)
        for _ in range(freq):
            db2.log_unknown_query(query, guess)

    final = db2.get_dashboard_stats()['overview']
    print(f"\n✅ Seeding complete!")
    print(f"   📊 Sessions:    {final['total_sessions']:,}")
    print(f"   💬 Messages:    {final['total_messages']:,}")
    print(f"   🎯 Avg Conf:    {final['avg_confidence']}%")
    print(f"   ⭐ Avg Rating:  {final['avg_rating']}")
    print(f"   📈 Intents tracked in analytics table")


if __name__ == '__main__':
    seed_database()
