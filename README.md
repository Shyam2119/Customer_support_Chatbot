# 🤖 Intelligent Customer Support Chatbot

A full-stack NLP-powered customer support chatbot built with **Python**, **TensorFlow**, **Flask**, **NLTK**, and **SQLite**.

---

## ✨ Features

### 🧠 NLP & Machine Learning
- **Intent classification** across 18 categories using a deep neural network (TensorFlow/Keras)
- **Bag-of-words** vectorization with NLTK tokenization + lemmatization
- **Multi-layer DNN** with BatchNorm, Dropout, and Adam optimizer
- **Confidence thresholding** — falls back gracefully for unknown inputs
- **Entity extraction** — order numbers, emails, phone numbers, dollar amounts, dates
- **Sentiment analysis** — positive/negative/neutral + urgency detection
- **Smart response personalization** based on detected entities and sentiment

### 🌐 REST API (Flask)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/session` | Create new chat session |
| POST | `/api/chat/message` | Send message, get AI response |
| GET | `/api/chat/history/<id>` | Full conversation history |
| POST | `/api/chat/feedback` | Submit 1-5 star feedback |
| GET | `/api/analytics/dashboard` | Full analytics dashboard data |
| GET | `/api/analytics/intents` | Per-intent performance metrics |
| GET | `/api/analytics/conversations` | Recent conversations list |
| GET | `/api/analytics/unknown-queries` | Unrecognized query analysis |
| GET | `/api/analytics/model-info` | Model metadata |
| GET | `/api/analytics/export` | Export conversation logs |
| POST | `/api/admin/reload-model` | Hot-reload model (API key required) |
| POST | `/api/admin/test-prediction` | Test prediction without logging |
| GET | `/health` | Health check |

### 💾 Database (SQLite)
- **5,000+ conversation capacity** with full conversation logs
- **7 normalized tables**: sessions, conversation_logs, feedback, intent_analytics, model_performance, escalations, unknown_queries
- **WAL mode** for concurrent access
- **Automatic indexing** for fast queries
- **Daily analytics aggregation** per intent
- **Export functionality** with date range filtering

### 📊 Analytics Dashboard
- Real-time metrics: sessions, messages, avg confidence, ratings
- Intent distribution bar chart
- Daily message volume line chart
- Sentiment breakdown donut chart
- Confidence distribution histogram
- Per-intent performance table with confidence bars
- Recent conversations table with escalation status
- Unknown query analysis for model improvement
- Model training metadata display
- Auto-refresh every 30 seconds

### 💬 Chat UI
- **4 switchable themes** — Dark 🌑, Light ☀️, Purple 💜, Ocean 🌊 (persisted across pages)
- Beautiful gradient-styled interface with Inter font
- Animated typing indicator with bouncing dots
- Real-time intent + confidence display bar
- Sentiment chips on bot messages (positive / negative / neutral / urgent)
- Entity extraction pills (order numbers, emails, phone, dollar amounts)
- 8 quick-reply shortcut buttons for common queries
- 👍/👎 micro-feedback on every bot message
- 5-star rating modal with optional free-text comment
- Session persistence via `sessionStorage` + history reload
- Mobile-responsive layout

---

## 🏗️ Project Structure

```
customer_support_chatbot/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── chatbot_engine.py    # NLP inference engine
│   ├── database.py          # SQLite database manager
│   └── routes.py            # All API routes
├── data/
│   ├── intents.json         # 18 intent categories (270+ patterns)
│   └── chatbot.db           # SQLite database (auto-created)
├── models/
│   ├── chatbot_model.h5     # Trained TensorFlow model
│   ├── words.pkl            # Vocabulary pickle
│   ├── classes.pkl          # Intent classes pickle
│   ├── training_metadata.json
│   └── training_history.json
├── scripts/
│   ├── train_model.py       # Full TensorFlow training pipeline
│   ├── quick_train.py       # Setup with fallback keyword model
│   └── seed_data.py         # Seed demo data
├── templates/
│   ├── index.html           # Chat UI
│   └── dashboard.html       # Analytics dashboard
├── tests/
│   └── test_chatbot.py      # Comprehensive test suite (40+ tests)
├── logs/                    # Application logs
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
└── run.py                   # Application entry point
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. One-Shot Setup (model + database)
```bash
# Builds keyword model + seeds 5,200+ conversation logs — no TensorFlow required
python setup.py
```

### 3. Start the Server
```bash
python run.py
```

### 4. Open the App
| URL | Description |
|-----|-------------|
| http://localhost:5000 | 💬 Chat UI (4 themes) |
| http://localhost:5000/dashboard | 📊 Analytics Dashboard |
| http://localhost:5000/health | ✅ Health check JSON |

### Optional — Train the Full TensorFlow Neural Network
```bash
# Higher accuracy (~97-99%) but requires TF + GPU (~5-10 min)
python scripts/train_model.py
```
Once `models/chatbot_model.h5` is created, the engine auto-detects it on next start
and switches to TF inference automatically.

> **Note:** The keyword fallback model is used by default until the `.h5` file exists.

---

## 🎯 Intent Categories (18)

| Intent | Description |
|--------|-------------|
| `greeting` | Hello, hi, hey |
| `goodbye` | Bye, farewell |
| `order_status` | Track order, shipment status |
| `return_refund` | Returns, refunds, exchanges |
| `product_inquiry` | Product details, availability |
| `payment_issues` | Failed payments, billing |
| `account_help` | Login, password, profile |
| `shipping_info` | Rates, options, delivery time |
| `technical_support` | App issues, bugs, errors |
| `pricing` | Cost, discounts, plans |
| `complaint` | Negative feedback, escalation |
| `cancellation` | Cancel orders/subscriptions |
| `promotions` | Deals, coupons, sales |
| `feedback` | Reviews, ratings |
| `business_hours` | Support availability |
| `human_agent` | Escalate to human |
| `warranty` | Product warranties, claims |
| `thanks` | Gratitude responses |

---

## 🧪 Running Tests
```bash
python -m pytest tests/ -v
# or
python tests/test_chatbot.py
```

Tests cover:
- Entity extraction (orders, emails, phones, amounts)
- Sentiment analysis (positive/negative/neutral/urgent)
- Database CRUD operations
- All Flask API endpoints
- Intent data validation
- Model loading

---

## 📡 API Examples

**Bash / curl**
```bash
# Create session
curl -X POST http://localhost:5000/api/chat/session

# Send message
curl -X POST http://localhost:5000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SID", "message": "Where is my order?"}'

# Analytics
curl http://localhost:5000/api/analytics/dashboard

# Admin reload
curl -X POST http://localhost:5000/api/admin/reload-model \
  -H "X-API-Key: demo-admin-key-12345"
```

**PowerShell**
```powershell
# Create session & send message
$s = Invoke-RestMethod -Uri http://localhost:5000/api/chat/session -Method POST
$sid = $s.session_id

Invoke-RestMethod -Uri http://localhost:5000/api/chat/message -Method POST `
  -ContentType "application/json" `
  -Body "{`"session_id`":`"$sid`",`"message`":`"Where is my order?`"}"
```

---

## 🔧 Model Architecture

```
Input (vocab_size)
    → Dense(256, relu) + BatchNorm + Dropout(0.4)
    → Dense(128, relu) + BatchNorm + Dropout(0.3)
    → Dense(64, relu)  + BatchNorm + Dropout(0.2)
    → Dense(num_classes, softmax)
```

- **Optimizer**: Adam (lr=0.001) with ReduceLROnPlateau
- **Loss**: Categorical Cross-Entropy
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Training Accuracy**: ~97-99% on training data

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and update:
```
SECRET_KEY=your-secret-key
ADMIN_API_KEY=your-admin-key
DEBUG=false  # for production
PORT=5000
```

---

## 🚢 Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Avg response time | 80–400 ms |
| Keyword model inference | < 10 ms |
| TF model inference | < 50 ms |
| DB seeded rows | 5,200+ |
| Concurrent sessions | ✅ WAL mode |

---

## 🎨 Themes

Both the chat page and the analytics dashboard share **4 built-in themes** via a `🎨 Theme` dropdown.  
The selected theme persists across page reloads and carries between both pages via `localStorage`.

| Theme | Style |
|-------|-------|
| 🌑 Dark | GitHub-dark blue/purple |
| ☀️ Light | Clean blue on white |
| 💜 Purple | Deep violet glassmorphism |
| 🌊 Ocean | Deep sea cyan/teal |

---

## 🐛 Known Issues Fixed

- **Connection error on first message** — Fixed: `import tensorflow` was blocking Flask threads  
  for 10–15 min when no `.h5` model existed. Now guarded behind a file-existence check.
