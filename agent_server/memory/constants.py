"""Memory-specific constants. Lakebase config imported from core.config."""

from core.config import settings

# ── Lakebase connection (from core.config, not duplicated) ──
LAKEBASE_INSTANCE_NAME = settings.LAKEBASE_INSTANCE_NAME
EMBEDDING_ENDPOINT = settings.EMBEDDING_ENDPOINT
EMBEDDING_DIMS = settings.EMBEDDING_DIMS

# ── Known metric names for extraction ──
KNOWN_METRICS = {
    "revenue", "click_rate", "click rate", "open_rate", "open rate",
    "unsubscribe_rate", "unsub rate", "unsubscribe", "unsubs",
    "conversion", "conversion_rate", "delivered", "delivery_rate",
    "delivery rate", "sent", "bounce", "bounce_rate", "bounce rate",
    "engagement", "performance", "roi",
}

# ── Normalize metric variants to canonical form ──
METRIC_CANONICAL = {
    "click rate": "click_rate",
    "open rate": "open_rate",
    "unsub rate": "unsubscribe_rate",
    "unsubscribe": "unsubscribe_rate",
    "unsubs": "unsubscribe_rate",
    "delivery rate": "delivery_rate",
    "bounce rate": "bounce_rate",
    "conversion_rate": "conversion",
}

# ── Limits ──
MAX_RECENT_QUERIES = 5
MAX_EPISODES_PER_CLIENT = 50
MAX_EPISODE_SEARCH_RESULTS = 3

# ── Channel names for extraction ──
CHANNEL_NAMES = ["email", "whatsapp", "sms", "apn", "bpn", "push", "in-app", "web"]
