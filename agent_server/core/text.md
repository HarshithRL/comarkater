START → supervisor ─┬→ greeting → END
                    ├→ clarification → END
                    └→ planner →  <- capability_router ─┬→ insight_agent (subgraph)
                                                    ├→ content_agent (subgraph)
                                                    └→ seo_agent (subgraph)
                                                             ↓
                                                        synthesizer
                                                             ↓
                                                           format