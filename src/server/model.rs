use crate::models::{
    Scorable,
    ncf::NeuMFConfig,
    deepfm::DeepFMConfig,
    gmf::GMFConfig
};
use burn::{
    backend::NdArray,
    module::Module,
    record::CompactRecorder,
    tensor::backend::Backend,
};
use super::state::Settings;

type B = NdArray<f32>;

pub fn load_model(
    settings: &Settings,
    device: &<B as Backend>::Device,
) -> anyhow::Result<Box<dyn Scorable<B> + Send>> {
    match settings.model_type.as_str() {
        "neumf" => {
            let config = NeuMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                gmf_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers.clone(),
                mlp_embed_dim: settings.mlp_embed_dim,
            };

            let model = config
                .init::<B>(device)
                .load_file(&settings.model, &CompactRecorder::new(), device)?;

            Ok(Box::new(model))
        }

        "deepfm" => {
            let config = DeepFMConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers.clone(),
            };

            let model = config
                .init::<B>(device)
                .load_file(&settings.model, &CompactRecorder::new(), device)?;

            Ok(Box::new(model))
        }

        "gmf" => {
            let config = GMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
            };

            let model = config
                .init::<B>(device)
                .load_file(&settings.model, &CompactRecorder::new(), device)?;

            Ok(Box::new(model))
        }

        _ => anyhow::bail!("unknown model_type: {}", settings.model_type),
    }
}
