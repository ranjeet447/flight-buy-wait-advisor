"""feature_store/definitions.py

Feast Entity and FeatureView definitions.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

# Entity: route_id (e.g., Delhi_Mumbai_Economy)
route_entity = Entity(
    name="route_id",
    value_type=String,
    description="Unique identifier for airline route and travel class",
)

# Source: Processed Parquet file
route_stats_source = FileSource(
    path="feature_store/data/route_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature View
route_features_view = FeatureView(
    name="route_features",
    entities=[route_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="avg_route_price", dtype=Float64),
        Field(name="min_route_price", dtype=Float64),
        Field(name="max_route_price", dtype=Float64),
        Field(name="std_route_price", dtype=Float64),
        Field(name="avg_duration", dtype=Float64),
    ],
    online=True,
    source=route_stats_source,
)