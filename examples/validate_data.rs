/// Smoke-test the data pipeline against the actual Myket CSV.
///
/// Usage:
///   cargo run --example validate_data
///   APP_data_path=data/movielens.csv cargo run --example validate_data
use burn_recsys::data::{NegativeSampler, PolarsDataset, RecsysDataset};
use serde::Deserialize;
use rand::SeedableRng;

#[derive(Deserialize, Debug)]
struct DataSettings {
    data_path: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Load configuration
    let builder = config::Config::builder()
        .add_source(config::File::with_name("config/validate_data.toml"))
        .add_source(config::Environment::with_prefix("APP")
            .try_parsing(true)
            .separator("__"));
    
    let config_built = builder.build()?;
    let settings: DataSettings = config_built.try_deserialize()?;

    println!("Configuration: {:?}", settings);
    println!("Loading {}...", settings.data_path);
    let ds = PolarsDataset::myket(&settings.data_path)?;

    println!("num_users   : {}", ds.num_users());
    println!("num_items   : {}", ds.num_items());
    println!("interactions: {}", ds.interactions().len());

    println!("\nSample interactions (first 5):");
    for &(u, i) in ds.interactions().iter().take(5) {
        println!("  user={u:>6}  item={i:>6}");
    }

    let (train, val) = ds.leave_one_out();
    println!("\ntrain : {} interactions", train.interactions().len());
    println!("val   : {} pairs (leave-one-out)", val.len());

    let sampler = NegativeSampler::new(ds.interactions(), ds.num_items(), 4);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let negs = sampler.sample(0, &mut rng);
    println!("\nNeg samples for user=0: {:?}", negs);

    println!("\nData pipeline OK");
    Ok(())
}
