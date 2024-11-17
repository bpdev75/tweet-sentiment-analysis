import logging
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
from opencensus.trace import config_integration
from opencensus.ext.azure.metrics_exporter import MetricsExporter
from opencensus.metrics.transport import get_exporter_thread
from opencensus.stats import stats
from opencensus.stats import view
from opencensus.stats.aggregation import LastValueAggregation
from opencensus.stats.measure import MeasureFloat
from opencensus.tags import TagMap

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

        # Configuration Azure
        exporter = MetricsExporter(connection_string=f"InstrumentationKey={instrumentationKey}")
        
        # Declare the accuracy measure
        self.accuracy_measure = MeasureFloat("accuracy", "Ratio de tweets correctement prédits", "ratio")
        accuracy_view = view.View(
            "accuracy_view",
            "Accuracy des prédictions",
            [],
            self.accuracy_measure,
            LastValueAggregation(),
        )
        monitor_stats = stats.Stats()
        monitor_stats.view_manager.register_exporter(exporter)
        monitor_stats.view_manager.register_view(accuracy_view)
        self.measure_map = monitor_stats.stats_recorder.new_measurement_map()

    def trace_feedback(self, tweet: str, predicted_sentiment: str, is_correct: bool):
        self.logger.info(
            "User feedback received",
            extra={
                "custom_dimensions": {
                    "tweet": tweet,
                    "predicted_sentiment": predicted_sentiment,
                    "feedback": "correct" if is_correct else "incorrect",
                }
            }
        )
        if is_correct:
            self.correct_predictions += 1
        self.total_predictions += 1

        # Calculate accuracy
        accuracy = self.correct_predictions / self.total_predictions
        
        # Record the accuracy
        self.measure_map.measure_float_put(self.accuracy_measure, accuracy)
        self.measure_map.record(TagMap())

    def logError(self, error):
        self.logger.error(error, exc_info=True)