"""Long-term memory manager for CoMarketer.

Provides per-client memory storage via AsyncDatabricksStore (Lakebase).
Primary key: client_scope — the Databricks secret scope name that identifies
each client tenant (e.g., "igp", "pepejeans", "crocs").

Two storage patterns:
  1. Client Profile — single document, store.aget(), ~30-50ms
  2. Episode Log — multiple documents, store.asearch(), ~100-200ms

Authentication: Uses the Databricks App's own identity (auto-injected by runtime).
All clients share one AsyncDatabricksStore instance — data is isolated by
client_scope namespace, not by separate connections.

All methods are safe to call from any context — failures are logged,
never raised. The agent continues without LTM if Lakebase is unavailable.
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

import mlflow
from databricks_langchain import AsyncDatabricksStore

from memory.constants import (
    LAKEBASE_INSTANCE_NAME,
    EMBEDDING_ENDPOINT,
    EMBEDDING_DIMS,
    MAX_RECENT_QUERIES,
    MAX_EPISODE_SEARCH_RESULTS,
    MAX_EPISODES_PER_CLIENT,
)

logger = logging.getLogger(__name__)


class LTMManager:
    """Long-term memory manager with per-client_scope namespace isolation.

    Single AsyncDatabricksStore instance shared across all clients, authenticated
    via the Databricks App's runtime identity. Data isolation is handled by
    namespace prefixes: ("client", "{client_scope}", "profile"|"episodes").
    """

    def __init__(
        self,
        instance_name: str = LAKEBASE_INSTANCE_NAME,
        embedding_endpoint: str = EMBEDDING_ENDPOINT,
        embedding_dims: int = EMBEDDING_DIMS,
    ):
        self.instance_name = instance_name
        self.embedding_endpoint = embedding_endpoint
        self.embedding_dims = embedding_dims
        self._store: Optional[AsyncDatabricksStore] = None
        logger.info(f"LTM: Manager created (instance={instance_name})")

    def attach_store(self, store: AsyncDatabricksStore) -> None:
        """Attach a pre-opened AsyncDatabricksStore.

        Called from the startup hook after the store's async context has been
        entered (pool open) and setup() has been awaited (tables created). From
        this point on, LTM read/write calls are served without per-request init.
        """
        self._store = store
        logger.info("LTM: store attached — ready to serve reads and writes")

    async def _get_store(self) -> Optional[AsyncDatabricksStore]:
        """Return the attached store, or None if LTM init failed/was skipped.

        LTM is optional — callers treat None as 'unavailable' and degrade
        gracefully without raising.
        """
        return self._store

    # ── Namespace helpers ──

    def _profile_ns(self, client_scope: str) -> tuple:
        """Namespace for client profile: ("client", "igp", "profile")"""
        return ("client", str(client_scope), "profile")

    def _episodes_ns(self, client_scope: str) -> tuple:
        """Namespace for episode log: ("client", "igp", "episodes")"""
        return ("client", str(client_scope), "episodes")

    # ── READ: Client Profile (single get, ~30-50ms) ──

    async def get_client_profile(self, client_scope: str) -> Dict:
        """Load the full client profile in ONE store.aget() call.

        Returns empty dict structure if no profile exists or on error.
        Called on every non-greeting request after ack is yielded.
        """
        store = await self._get_store()
        if store is None:
            logger.warning(f"LTM READ: store unavailable, returning empty profile | scope={client_scope}")
            return self._empty_profile()
        with mlflow.start_span(name="ltm_read_profile") as span:
            span.set_attributes({"ltm.client_scope": client_scope})
            try:
                result = await store.aget(self._profile_ns(client_scope), "memory")
                if result and result.value:
                    span.set_attributes({
                        "ltm.status": "found",
                        "ltm.total_queries": result.value.get("total_queries", 0),
                    })
                    return result.value
                span.set_attributes({"ltm.status": "not_found"})
                return self._empty_profile()
            except Exception as e:
                logger.error(f"LTM READ: Profile failed for scope={client_scope}: {e}")
                span.set_attributes({"ltm.status": "error", "ltm.error": str(e)[:200]})
                return self._empty_profile()

    # ── READ: Episodes (semantic search, ~100-200ms) ──

    async def search_past_episodes(self, client_scope: str, query: str) -> List[Dict]:
        """Search past episodes by semantic similarity to current query.

        Only called for optimization/complex_analysis queries — NOT for
        greetings or simple lookups.

        Returns list of episode value dicts (max 3).
        """
        store = await self._get_store()
        if store is None:
            logger.warning(f"LTM SEARCH: store unavailable, returning empty | scope={client_scope}")
            return []
        with mlflow.start_span(name="ltm_search_episodes") as span:
            span.set_attributes({"ltm.client_scope": client_scope, "ltm.query_len": len(query)})
            try:
                results = await store.asearch(
                    self._episodes_ns(client_scope),
                    query=query,
                    limit=MAX_EPISODE_SEARCH_RESULTS,
                )
                episodes = [item.value for item in results if item.value]
                span.set_attributes({"ltm.episodes_found": len(episodes)})
                return episodes
            except Exception as e:
                logger.error(f"LTM SEARCH: Episodes failed for scope={client_scope}: {e}")
                span.set_attributes({"ltm.status": "error"})
                return []

    # ── WRITE: Update Client Profile (read-modify-write, ~60-100ms) ──

    async def update_client_profile(
        self,
        client_scope: str,
        entities: Dict,
        user_query: str,
    ) -> bool:
        """Update client profile with extracted entities from current query.

        Called POST-YIELD — zero user-facing latency.
        Read-modify-write on single document.
        """
        store = await self._get_store()
        if store is None:
            logger.warning(f"LTM WRITE: store unavailable, skipping profile update | scope={client_scope}")
            return False
        with mlflow.start_span(name="ltm_write_profile") as span:
            span.set_attributes({"ltm.client_scope": client_scope})
            try:
                profile = await self.get_client_profile(client_scope)

                ch_freq = profile.get("channel_frequency", {})
                for ch in entities.get("channels", []):
                    ch_freq[ch] = ch_freq.get(ch, 0) + 1
                profile["channel_frequency"] = ch_freq

                m_freq = profile.get("metric_frequency", {})
                for m in entities.get("metrics", []):
                    m_freq[m] = m_freq.get(m, 0) + 1
                profile["metric_frequency"] = m_freq

                tp_freq = profile.get("time_period_frequency", {})
                for tp in entities.get("time_periods", []):
                    tp_freq[tp] = tp_freq.get(tp, 0) + 1
                profile["time_period_frequency"] = tp_freq

                recent = profile.get("recent_queries", [])
                recent.append(user_query[:200])
                if len(recent) > MAX_RECENT_QUERIES:
                    recent = recent[-MAX_RECENT_QUERIES:]
                profile["recent_queries"] = recent

                profile["last_updated"] = datetime.utcnow().isoformat() + "Z"
                profile["total_queries"] = profile.get("total_queries", 0) + 1

                await store.aput(self._profile_ns(client_scope), "memory", profile)

                span.set_attributes({
                    "ltm.channels_updated": len(entities.get("channels", [])),
                    "ltm.metrics_updated": len(entities.get("metrics", [])),
                    "ltm.total_queries": profile["total_queries"],
                })
                logger.info(f"LTM WRITE: Profile updated for scope={client_scope}")
                return True
            except Exception as e:
                logger.error(f"LTM WRITE: Profile failed for scope={client_scope}: {e}")
                span.set_attributes({"ltm.status": "error"})
                return False

    # ── WRITE: Save Episode (single put, ~30-50ms) ──

    async def save_episode(
        self,
        client_scope: str,
        query: str,
        agent_route: str,
        finding: str,
        channels_involved: List[str],
        metrics_involved: List[str],
        success: bool = True,
    ) -> bool:
        """Save a query-finding pair as an episodic memory.

        Called POST-YIELD — zero user-facing latency.
        Only called when a meaningful finding was extracted from the response.
        """
        store = await self._get_store()
        if store is None:
            logger.warning(f"LTM WRITE: store unavailable, skipping episode save | scope={client_scope}")
            return False
        with mlflow.start_span(name="ltm_write_episode") as span:
            span.set_attributes({"ltm.client_scope": client_scope})
            try:
                episode_id = f"insight_{uuid.uuid4().hex[:8]}"
                episode = {
                    "query": query[:200],
                    "agent_route": agent_route,
                    "finding": finding[:500],
                    "channels_involved": channels_involved,
                    "metrics_involved": metrics_involved,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                ns = self._episodes_ns(client_scope)
                await store.aput(ns, episode_id, episode)
                span.set_attributes({"ltm.episode_id": episode_id})
                logger.info(f"LTM WRITE: Episode {episode_id} saved for scope={client_scope}")

                await self._prune_old_episodes(store, ns)
                return True
            except Exception as e:
                logger.error(f"LTM WRITE: Episode failed for scope={client_scope}: {e}")
                return False

    # ── Cleanup ──

    async def _prune_old_episodes(self, store: AsyncDatabricksStore, namespace: tuple) -> None:
        """Delete oldest episodes beyond MAX_EPISODES_PER_CLIENT.

        Non-fatal — failures are logged silently.
        """
        try:
            all_eps = await store.asearch(namespace, query="", limit=MAX_EPISODES_PER_CLIENT + 20)
            if len(all_eps) <= MAX_EPISODES_PER_CLIENT:
                return

            sorted_eps = sorted(
                all_eps,
                key=lambda item: item.value.get("timestamp", "") if item.value else "",
            )
            to_delete = sorted_eps[: len(sorted_eps) - MAX_EPISODES_PER_CLIENT]
            for item in to_delete:
                await store.adelete(namespace, item.key)
            logger.info(f"LTM PRUNE: Deleted {len(to_delete)} old episodes from {namespace}")
        except Exception as e:
            logger.warning(f"LTM PRUNE: Failed for {namespace}: {e}")

    # ── Helper ──

    @staticmethod
    def _empty_profile() -> Dict:
        return {
            "channel_frequency": {},
            "metric_frequency": {},
            "time_period_frequency": {},
            "recent_queries": [],
            "last_updated": None,
            "total_queries": 0,
        }
