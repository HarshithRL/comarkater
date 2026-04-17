"""Loader and formatter for Campaign Insight Agent domain knowledge YAMLs.

Reads the six YAML files under ``domain_knowledge/`` once at construction time
and exposes typed lookup methods plus two prompt-context formatters
(``format_for_supervisor`` — condensed; ``format_for_subagent`` — full).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from agents.campaign_insight.contracts import PatternMatch, Recommendation

logger = logging.getLogger(__name__)

_YAML_FILES = {
    "metrics": "metrics.yaml",
    "interpretation": "interpretation.yaml",
    "constraints": "constraints.yaml",
    "recommendations": "recommendations.yaml",
    "intent": "intent.yaml",
    "domain_context": "domain_context.yaml",
    "audience": "audience_knowledge.yaml",
    "content": "content_knowledge.yaml",
}


class InsightAgentDomainKnowledge:
    """Central YAML loader for the Campaign Insight Agent."""

    def __init__(self, knowledge_dir: str | Path) -> None:
        self._dir = Path(knowledge_dir)
        self._data: dict[str, dict] = {}
        for key, filename in _YAML_FILES.items():
            path = self._dir / filename
            if not path.is_file():
                logger.warning("Domain knowledge file missing: %s", path)
                self._data[key] = {}
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    self._data[key] = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
                self._data[key] = {}

    # ------------------------------------------------------------------ metrics

    def _all_metrics(self) -> list[dict]:
        m = self._data.get("metrics", {})
        return list(m.get("base_metrics", []) or []) + list(m.get("derived_metrics", []) or [])

    def get_metric_definition(self, metric_name: str) -> Optional[dict]:
        """Return formula, denominator, unit, and channel_applicability for a metric."""
        for entry in self._all_metrics():
            if entry.get("name") == metric_name:
                return {
                    "name": entry.get("name"),
                    "display_name": entry.get("display_name", ""),
                    "formula": entry.get("formula", ""),
                    "numerator": entry.get("numerator", ""),
                    "denominator": entry.get("denominator", ""),
                    "unit": entry.get("unit", ""),
                    "channel_applicability": entry.get("channel_applicability", {}),
                    "notes": entry.get("notes", ""),
                }
        return None

    def get_metric_thresholds(self, metric_name: str) -> Optional[dict]:
        """Return threshold bands for ``metric_name`` from interpretation.yaml."""
        for rule in self._data.get("interpretation", {}).get("single_metric_rules", []) or []:
            if rule.get("metric") == metric_name:
                return rule.get("thresholds", {}) or {}
        return None

    def get_all_metric_names(self) -> list[str]:
        return [m.get("name", "") for m in self._all_metrics() if m.get("name")]

    # ------------------------------------------------------- pattern detection

    def _metric_level(self, metric: str, value: float) -> Optional[str]:
        """Bucket a metric value into ``high`` / ``low`` / ``mid`` using thresholds.

        Negative metrics (bounce/unsubscribe/suppression) invert the polarity:
        a high *value* is bad = semantic ``high``; a low value = semantic ``low``.
        """
        thresholds = self.get_metric_thresholds(metric) or {}
        if not thresholds:
            return None
        high_min: Optional[float] = None
        low_max: Optional[float] = None
        for label, band in thresholds.items():
            if not isinstance(band, dict):
                continue
            if label in ("excellent", "good") and "min" in band:
                high_min = band["min"] if high_min is None else min(high_min, band["min"])
            if label in ("critical", "bad", "warning", "low") and "max" in band:
                low_max = band["max"] if low_max is None else max(low_max, band["max"])
            if label in ("critical",) and "min" in band:  # negative metrics (high = bad)
                high_min = band["min"] if high_min is None else min(high_min, band["min"])
            if label in ("good",) and "max" in band:  # negative metrics (low = good)
                low_max = band["max"] if low_max is None else max(low_max, band["max"])
        if high_min is not None and value >= high_min:
            return "high"
        if low_max is not None and value <= low_max:
            return "low"
        return "mid"

    def detect_combination_patterns(self, metric_values: dict) -> list[PatternMatch]:
        """Match provided metric values against combination_patterns."""
        matches: list[PatternMatch] = []
        for pat in self._data.get("interpretation", {}).get("combination_patterns", []) or []:
            cond = pat.get("condition", {}) or {}
            highs = cond.get("high", []) or []
            lows = cond.get("low", []) or []
            if not (highs or lows):
                continue
            if any(m not in metric_values for m in highs + lows):
                continue
            ok = all(self._metric_level(m, metric_values[m]) == "high" for m in highs) and \
                 all(self._metric_level(m, metric_values[m]) == "low" for m in lows)
            if ok:
                matches.append(PatternMatch(
                    name=pat.get("name", ""),
                    description=pat.get("interpretation", ""),
                    likely_causes=list(pat.get("likely_causes", []) or []),
                    severity=pat.get("severity", ""),
                ))
        return matches

    # ------------------------------------------------------------------- trend

    def get_trend_interpretation(self, pattern: str, metric: str) -> dict:
        """Return polarity-aware trend interpretation for a pattern+metric pair."""
        for rule in self._data.get("interpretation", {}).get("trend_rules", []) or []:
            if rule.get("pattern") != pattern:
                continue
            depends = rule.get("interpretation_depends_on", {}) or {}
            pos = depends.get("positive_metrics", []) or []
            neg = depends.get("negative_metrics", []) or []
            polarity: Optional[str] = None
            if metric in pos:
                polarity = "good" if pattern == "increasing" else "bad" if pattern == "decreasing" else None
            elif metric in neg:
                polarity = "bad" if pattern == "increasing" else "good" if pattern == "decreasing" else None
            return {
                "pattern": pattern,
                "metric": metric,
                "description": rule.get("description", ""),
                "polarity": polarity,
                "interpretation": rule.get("interpretation", ""),
                "likely_causes": list(rule.get("likely_causes", []) or []),
                "notes": rule.get("notes", ""),
            }
        return {"pattern": pattern, "metric": metric, "polarity": None}

    # --------------------------------------------------------- recommendations

    def get_recommendations_for_pattern(self, pattern_name: str) -> list[Recommendation]:
        """Return Recommendations attached to a combination-pattern name."""
        for pat in self._data.get("interpretation", {}).get("combination_patterns", []) or []:
            if pat.get("name") != pattern_name:
                continue
            evidence = "; ".join(pat.get("likely_causes", []) or [])
            return [
                Recommendation(action=str(rec), evidence=evidence, source_pattern=pattern_name)
                for rec in pat.get("recommendations", []) or []
            ]
        return []

    # --------------------------------------------------------------- constraints

    def get_channel_constraints(self, channel: str) -> dict:
        """Merge channel constraints + channel context for a given channel."""
        merged: dict[str, Any] = {"channel": channel}
        for entry in self._data.get("constraints", {}).get("channel_constraints", []) or []:
            if isinstance(entry, dict) and entry.get("channel") == channel:
                merged.update({k: v for k, v in entry.items() if k != "channel"})
        for entry in self._data.get("domain_context", {}).get("channels", []) or []:
            if entry.get("code") == channel:
                merged.update({
                    "tracks_opens": entry.get("tracks_opens"),
                    "tracks_clicks": entry.get("tracks_clicks"),
                    "tracks_conversions": entry.get("tracks_conversions"),
                    "has_subject": entry.get("has_subject"),
                    "typical_ctr": entry.get("typical_ctr"),
                    "typical_open_rate": entry.get("typical_open_rate"),
                })
        return merged

    def get_channel_recommendations(self, channel: str, underperforming: bool) -> list[str]:
        block = self._data.get("recommendations", {}).get("channel_recommendations", {}) or {}
        entry = block.get(channel, {}) or {}
        key = "when_underperforming" if underperforming else "general"
        return list(entry.get(key, []) or [])

    def get_out_of_scope_rules(self) -> list[dict]:
        rules = self._data.get("constraints", {}).get("analysis_constraints", {}).get("cannot_answer", []) or []
        return [
            {"question": r.get("question", ""), "reason": r.get("reason", ""), "alternative": r.get("alternative", "")}
            for r in rules if isinstance(r, dict)
        ]

    # -------------------------------------------------------------------- intent

    def get_intent_taxonomy(self) -> list[dict]:
        return list(self._data.get("intent", {}).get("intents", []) or [])

    def get_response_pattern(self, intent_type: str) -> str:
        for intent in self.get_intent_taxonomy():
            if intent.get("name") == intent_type:
                return intent.get("response_pattern", "")
        return ""

    # -------------------------------------------------------------------- funnel

    def get_funnel_stages(self) -> list[dict]:
        stages = self._data.get("domain_context", {}).get("funnel", {}).get("stages", []) or []
        return [
            {
                "name": s.get("name", ""),
                "next": s.get("next", ""),
                "meaning": s.get("meaning", ""),
                "drop_reason": s.get("drop_reason", ""),
                "channel_exception": s.get("channel_exception", ""),
            }
            for s in stages if isinstance(s, dict)
        ]

    # ------------------------------------------------------------------- context

    def get_platform_context(self) -> dict:
        plat = self._data.get("domain_context", {}).get("platform", {}) or {}
        channels = [
            {
                "code": c.get("code"),
                "display_name": c.get("display_name"),
                "tracks_opens": c.get("tracks_opens"),
                "tracks_clicks": c.get("tracks_clicks"),
                "tracks_conversions": c.get("tracks_conversions"),
                "has_subject": c.get("has_subject"),
            }
            for c in self._data.get("domain_context", {}).get("channels", []) or []
        ]
        return {"platform": plat, "channels": channels}

    def get_minimum_volume_thresholds(self) -> dict:
        mv = self._data.get("constraints", {}).get("metric_constraints", {}).get("minimum_volume", {}) or {}
        return {
            "rate_ranking": mv.get("rate_ranking", ""),
            "top_campaigns": mv.get("top_campaigns", ""),
            "statistical": mv.get("statistical", ""),
            "reason": mv.get("reason", ""),
        }

    # ---------------------------------------------------------------- audience

    def get_audience_tag_categories(self) -> list[dict]:
        return list(self._data.get("audience", {}).get("tag_categories", []) or [])

    def get_audience_interpretation_rules(self) -> dict:
        return dict(self._data.get("audience", {}).get("interpretation_rules", {}) or {})

    def get_audience_diagnostic_patterns(self) -> list[dict]:
        return list(self._data.get("audience", {}).get("diagnostic_patterns", []) or [])

    def get_audience_nudge_triggers(self) -> dict:
        return dict(self._data.get("audience", {}).get("nudge_triggers", {}) or {})

    # ----------------------------------------------------------------- content

    def get_content_scores(self) -> list[dict]:
        return list(self._data.get("content", {}).get("content_scores", []) or [])

    def get_content_features(self) -> list[dict]:
        return list(self._data.get("content", {}).get("content_features", []) or [])

    def get_content_diagnostic_patterns(self) -> list[dict]:
        return list(self._data.get("content", {}).get("diagnostic_patterns", []) or [])

    def get_content_benchmark_rules(self) -> dict:
        return dict(self._data.get("content", {}).get("benchmark_rules", {}) or {})

    def get_content_default_metric(self) -> str:
        return "click"

    # ========================================================== prompt formatters

    def format_for_supervisor(self) -> str:
        """Condensed domain context for the fast supervisor model."""
        plat = self._data.get("domain_context", {}).get("platform", {}) or {}
        lines: list[str] = []
        lines.append("# DOMAIN CONTEXT (supervisor)")
        lines.append(f"Platform: {plat.get('name', '')} — {plat.get('type', '')}")
        lines.append(f"Scope: {plat.get('customer_lifecycle', {}).get('analysis_scope', '')}")
        lines.append("")
        lines.append("## Channels (code — display — tracking)")
        for c in self._data.get("domain_context", {}).get("channels", []) or []:
            lines.append(
                f"- {c.get('code')} | {c.get('display_name')} | "
                f"opens={c.get('tracks_opens')} clicks={c.get('tracks_clicks')} "
                f"conv={c.get('tracks_conversions')} subject={c.get('has_subject')}"
            )
        lines.append("")
        lines.append("## Metrics (names only — no formulas)")
        base = [m.get("name") for m in self._data.get("metrics", {}).get("base_metrics", []) or []]
        der = [m.get("name") for m in self._data.get("metrics", {}).get("derived_metrics", []) or []]
        lines.append(f"Base: {', '.join(base)}")
        lines.append(f"Derived: {', '.join(der)}")
        lines.append("")
        lines.append("## Channel applicability (metric → channels that DO NOT support it)")
        for m in self._all_metrics():
            app = m.get("channel_applicability") or {}
            unsupported = [ch for ch, ok in app.items() if ok is False]
            if unsupported:
                lines.append(f"- {m.get('name')}: not applicable for {', '.join(unsupported)}")
        lines.append("")
        lines.append("## Intent taxonomy")
        for intent in self.get_intent_taxonomy():
            examples = intent.get("examples", []) or []
            ex = f" — e.g. \"{examples[0]}\"" if examples else ""
            lines.append(
                f"- {intent.get('name')} ({intent.get('complexity', 'n/a')}): "
                f"{intent.get('description', '')}{ex}"
            )
        lines.append("")
        lines.append("## Channel constraints")
        for entry in self._data.get("constraints", {}).get("channel_constraints", []) or []:
            if isinstance(entry, dict) and entry.get("channel"):
                flag_bits = [k for k, v in entry.items() if v is True and k != "channel"]
                rule = entry.get("rule", "")
                lines.append(f"- {entry.get('channel')}: {', '.join(flag_bits) or '-'} | {rule}")
        lines.append("")
        lines.append("## Out of scope")
        for r in self.get_out_of_scope_rules():
            lines.append(f"- Q: {r['question']} — Reason: {r['reason']} — Alt: {r['alternative']}")
        return "\n".join(lines)

    def format_for_subagent(self) -> str:
        """Full domain reference for the reasoning subagent model."""
        parts: list[str] = [self.format_for_supervisor(), ""]
        parts.append("## Metric formulas")
        for m in self._data.get("metrics", {}).get("derived_metrics", []) or []:
            parts.append(
                f"- {m.get('name')}: {m.get('formula', '')} "
                f"(num={m.get('numerator', '')}, den={m.get('denominator', '')}, unit={m.get('unit', '')})"
            )
        parts.append("")
        parts.append("## Interpretation thresholds")
        for rule in self._data.get("interpretation", {}).get("single_metric_rules", []) or []:
            bands = rule.get("thresholds", {}) or {}
            band_strs = []
            for label, band in bands.items():
                if isinstance(band, dict):
                    band_strs.append(f"{label}={band}")
            parts.append(f"- {rule.get('metric')}: " + "; ".join(band_strs))
        parts.append("")
        parts.append("## Combination patterns")
        for pat in self._data.get("interpretation", {}).get("combination_patterns", []) or []:
            cond = pat.get("condition", {}) or {}
            parts.append(
                f"- {pat.get('name')}: high={cond.get('high', [])} low={cond.get('low', [])} "
                f"→ {pat.get('interpretation', '')}"
            )
        parts.append("")
        parts.append("## Trend rules")
        for rule in self._data.get("interpretation", {}).get("trend_rules", []) or []:
            depends = rule.get("interpretation_depends_on", {}) or {}
            parts.append(
                f"- {rule.get('pattern')}: {rule.get('description', '')} "
                f"(positive={depends.get('positive_metrics', [])}, negative={depends.get('negative_metrics', [])})"
            )
        parts.append("")
        parts.append("## Recommendation triggers")
        for pat in self._data.get("recommendations", {}).get("recommendation_patterns", []) or []:
            trig = pat.get("trigger", {}) or {}
            parts.append(
                f"- trigger: metric={trig.get('metric')} cond={trig.get('condition')} "
                f"→ {pat.get('diagnosis', '')} (severity={pat.get('severity', '')})"
            )
        parts.append("")
        parts.append("## Minimum volume thresholds")
        for k, v in self.get_minimum_volume_thresholds().items():
            parts.append(f"- {k}: {v}")
        parts.append("")
        parts.append("## Channel-specific recommendations")
        block = self._data.get("recommendations", {}).get("channel_recommendations", {}) or {}
        for ch, recs in block.items():
            parts.append(f"- {ch} general: {'; '.join(recs.get('general', []) or [])}")
            parts.append(f"- {ch} when_underperforming: {'; '.join(recs.get('when_underperforming', []) or [])}")
        parts.append("")
        parts.append("## Response template")
        tmpl = self._data.get("recommendations", {}).get("response_template", {}) or {}
        for section in tmpl.get("structure", []) or []:
            parts.append(f"- {section.get('section')}: {section.get('content')}")

        # ── Audience domain knowledge ──
        tag_cats = self.get_audience_tag_categories()
        aud_rules = self.get_audience_interpretation_rules()
        nudge = self.get_audience_nudge_triggers()
        if tag_cats or aud_rules or nudge:
            parts.append("")
            parts.append("## Audience tag categories")
            for cat in tag_cats:
                values = ", ".join(str(v) for v in (cat.get("values") or []))
                parts.append(
                    f"- {cat.get('name')} ({cat.get('display_name', '')}): "
                    f"values=[{values}] — {cat.get('key_insight', '')}"
                )

        if aud_rules:
            parts.append("")
            parts.append("## Audience interpretation rules")
            for rule_name, rule in aud_rules.items():
                if not isinstance(rule, dict):
                    continue
                thresholds = rule.get("thresholds", {}) or {}
                th_bits = []
                for label, band in thresholds.items():
                    if isinstance(band, dict):
                        th_bits.append(f"{label}={ {k: v for k, v in band.items() if k != 'label'} }")
                parts.append(
                    f"- {rule_name}: {rule.get('description', '')} "
                    f"(formula={rule.get('formula', rule.get('condition', ''))}) "
                    f"thresholds: {'; '.join(th_bits) if th_bits else '-'}"
                )

        if nudge:
            parts.append("")
            parts.append("## Audience nudge triggers")
            for cond in nudge.get("conditions", []) or []:
                parts.append(
                    f"- when {cond.get('trigger', '')}: \"{cond.get('nudge', '')}\""
                )

        # ── Content domain knowledge ──
        scores = self.get_content_scores()
        features = self.get_content_features()
        if scores or features:
            parts.append("")
            parts.append(f"## Content default metric: {self.get_content_default_metric()}")

        if scores:
            parts.append("")
            parts.append("## Content score interpretation")
            for s in scores:
                interp = s.get("interpretation") or {}
                band_bits = []
                for label, band in interp.items():
                    if isinstance(band, dict):
                        band_bits.append(f"{label}={ {k: v for k, v in band.items() if k != 'label'} }")
                line = (
                    f"- {s.get('name')} ({s.get('display_name', '')}): "
                    f"{'; '.join(band_bits) if band_bits else s.get('description', '')}"
                )
                parts.append(line)

        if features:
            parts.append("")
            parts.append("## Content feature interpretation rules")
            for f in features:
                interp = f.get("interpretation") or {}
                interp_bits = []
                if isinstance(interp, dict):
                    for label, band in interp.items():
                        if isinstance(band, dict):
                            interp_bits.append(f"{label}={ {k: v for k, v in band.items() if k != 'label'} }")
                        else:
                            interp_bits.append(f"{label}: {band}")
                parts.append(
                    f"- {f.get('name')} ({f.get('display_name', '')}, type={f.get('type', '')}): "
                    f"{'; '.join(interp_bits) if interp_bits else f.get('description', '')}"
                )

        return "\n".join(parts)
