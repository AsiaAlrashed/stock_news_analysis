from mage_ai.orchestration.triggers.api import trigger_pipeline_run

pipeline = ["fetch_news", "predict_sentiment", "rank_system"]

trigger_pipeline_run(pipeline)
