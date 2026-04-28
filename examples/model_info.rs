/// Inspect model parameters and architecture.
///
/// Usage:
///   cargo run --example model_info
///   APP_model_type=deepfm cargo run --example model_info
use burn::backend::NdArray;
use burn_recsys::server::Settings;
use burn_recsys::models::ncf::NeuMFConfig;
use burn_recsys::models::deepfm::DeepFMConfig;
use burn_recsys::models::gmf::GMFConfig;

type B = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Load configuration
    let builder = config::Config::builder()
        .add_source(config::File::with_name("config/model_info.toml"))
        .add_source(config::Environment::with_prefix("APP")
            .try_parsing(true)
            .separator("__"));
    
    let config_built = builder.build()?;
    let settings: Settings = config_built.try_deserialize()?;

    println!("Configuration: {:?}", settings);
    let device = Default::default();

    println!("=== Model Architecture Info ===");
    
    match settings.model_type.as_str() {
        "neumf" => {
            let config = NeuMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                gmf_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers,
                mlp_embed_dim: settings.mlp_embed_dim,
            };
            let model = config.init::<B>(&device);
            println!("Model Type: NeuMF");
            println!("Parameters: {}", model.num_params());
        }
        "deepfm" => {
            let config = DeepFMConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers,
            };
            let model = config.init::<B>(&device);
            println!("Model Type: DeepFM");
            println!("Parameters: {}", model.num_params());
        }
        "gmf" => {
            let config = GMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
            };
            let model = config.init::<B>(&device);
            println!("Model Type: GMF");
            println!("Parameters: {}", model.num_params());
        }
        _ => println!("Unknown model_type: {}", settings.model_type),
    }

    Ok(())
}
