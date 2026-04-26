/// OpenTelemetry + tracing setup.
///
/// Call `init()` at the start of main. Spans and metrics are exported
/// to stdout in OTLP-compatible JSON — pipe to Jaeger/Prometheus later.
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_stdout::MetricsExporter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Metrics available throughout the application.
pub struct Metrics {
    pub recommend_requests:   Counter<u64>,
    pub recommend_latency_ms: Histogram<f64>,
    pub train_epoch_loss:     Histogram<f64>,
    pub data_load_rows:       Counter<u64>,
}

impl Metrics {
    fn new(meter: &Meter) -> Self {
        Self {
            recommend_requests: meter
                .u64_counter("recsys.recommend.requests")
                .with_description("Total recommendation requests served")
                .init(),
            recommend_latency_ms: meter
                .f64_histogram("recsys.recommend.latency_ms")
                .with_description("Recommendation request latency in milliseconds")
                .init(),
            train_epoch_loss: meter
                .f64_histogram("recsys.train.epoch_loss")
                .with_description("Training loss per epoch")
                .init(),
            data_load_rows: meter
                .u64_counter("recsys.data.rows_loaded")
                .with_description("Total rows loaded from dataset")
                .init(),
        }
    }
}

/// Initialize tracing to stdout (call once at startup, non-async context).
/// Returns a `Metrics` handle backed by the global OTel meter.
///
/// Note: for the full OTel metrics pipeline with async export, call
/// `init_async()` inside a Tokio runtime instead.
pub fn init_logging() {
    let _ = tracing_subscriber::registry()
        .with(
            EnvFilter::from_default_env()
                .add_directive("burn_recsys=info".parse().unwrap())
                .add_directive("info".parse().unwrap()),
        )
        .with(tracing_subscriber::fmt::layer().compact())
        .try_init();
}

/// Initialize full OTel pipeline (requires Tokio runtime).
pub fn init_metrics() -> Metrics {
    let exporter = MetricsExporter::default();
    let reader = PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    global::set_meter_provider(provider);

    let meter = global::meter("burn_recsys");
    Metrics::new(&meter)
}

/// Record a recommendation request.
pub fn record_request(metrics: &Metrics, latency_ms: f64, model: &str) {
    let labels = [KeyValue::new("model", model.to_string())];
    metrics.recommend_requests.add(1, &labels);
    metrics.recommend_latency_ms.record(latency_ms, &labels);
}
