import logging
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace import Tracer
from opencensus.trace import config_integration

class Monitoring:

    def __init__(self):
        instrumentationKey = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")

        # Configurer intégration pour FastAPI
        config_integration.trace_integrations(['logging', 'requests'])

        # Configurer le logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={instrumentationKey}"))
        self.logger.setLevel(logging.INFO)

        # Configurer le tracer
        self.tracer = Tracer(
            exporter=AzureExporter(connection_string=f"InstrumentationKey={instrumentationKey}"),
            sampler=ProbabilitySampler(1.0)
        )
        self.correct_predictions = 0
        self.total_predictions = 0

    def updateAccuracy(self, tweet: str, predicted_sentiment: str, user_feedback: bool):
        if not user_feedback:
            self.logger.warning("Incorrectly predicted tweet sentiment", extra={
                "custom_dimensions": {
                    "tweet_text": tweet,
                    "predicted_sentiment": predicted_sentiment
                }
            })
        else:
            self.correct_predictions += 1
        self.total_predictions += 1
