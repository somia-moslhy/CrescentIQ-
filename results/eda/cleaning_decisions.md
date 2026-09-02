# Cleaning Decisions
- MarkDown1-5 missing -> filled with 0 (no promo that week, domain fact).
- CPI/Unemployment missing -> forward-filled per store (slow-changing indicators).
- Negative Weekly_Sales (1285 rows) -> kept, represent real returns.
- IQR outliers (35521 rows) -> kept, represent real holiday demand spikes.
