/// Smoke-test the data pipeline against the actual Myket CSV.
/// Run: cargo run --example validate_data
/// Run: cargo run --example validate_data -- --data data/myket.csv
use burn_recsys::data::{NegativeSampler, PolarsDataset, RecsysDataset};
use clap::Parser;
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[command(about = "Validate the data pipeline against a CSV file")]
struct Args {
    /// Path to the dataset CSV (Myket format: user_id, app_name, timestamp)
    #[arg(long, default_value = "data/myket.csv")]
    data: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Loading {}...", args.data);
    let ds = PolarsDataset::myket(&args.data)?;

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
