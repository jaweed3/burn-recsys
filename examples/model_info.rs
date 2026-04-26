/// Print model architecture info and param counts for GMF and NeuMF.
/// Run: cargo run --example model_info
use burn::backend::NdArray;
use burn_recsys::models::{
    gmf::GMFConfig,
    ncf::NeuMFConfig,
};

type B = NdArray<f32>;

fn main() {
    // Myket dataset dimensions
    let num_users = 10_000;
    let num_items = 7_988;
    let device = Default::default();

    // GMF
    let gmf_cfg = GMFConfig { num_users, num_items, embedding_dim: 64 };
    let gmf = gmf_cfg.init::<B>(&device);
    let gmf_params = gmf.num_params();

    // NeuMF
    let ncf_cfg = NeuMFConfig {
        num_users,
        num_items,
        gmf_dim: 64,
        mlp_layers: vec![128, 64, 32, 16],
        mlp_embed_dim: 64,
    };
    let ncf = ncf_cfg.init::<B>(&device);
    let ncf_params = ncf.num_params();

    println!("=== Model Architecture ===");
    println!();
    println!("Dataset  : Myket ({num_users} users, {num_items} items)");
    println!();
    println!("GMF");
    println!("  user emb   : {num_users} × 64");
    println!("  item emb   : {num_items} × 64");
    println!("  output     : 64 → 1");
    println!("  params     : {gmf_params:>10}");
    println!();
    println!("NeuMF (GMF path + MLP path)");
    println!("  GMF user/item emb : {num_users} × 64  |  {num_items} × 64");
    println!("  MLP user/item emb : {num_users} × 64  |  {num_items} × 64");
    println!("  MLP layers        : 128→64→32→16");
    println!("  output            : (64+16) → 1");
    println!("  params            : {ncf_params:>10}");
}
