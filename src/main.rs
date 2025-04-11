mod token_output_stream;
use std::io::Write;
use tokenizers::Tokenizer;

use candle_core::quantized::gguf_file;
use candle_core::{Tensor, Device};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;

use promkit::preset::readline::Readline;

#[derive(Debug)]
struct ModelConfig {
    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// Process prompt elements separately.
    split_prompt: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            seed: 299792458,
            split_prompt: false,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            top_p: None,
            top_k: None
        }
    }
}

impl ModelConfig {
    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";
        let api = api.model(repo.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<Qwen2> {
        let (repo, filename, revision) = (
            "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            "main",
        );
        let api = hf_hub::api::sync::Api::new()?;
        let model_path = api.repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                revision.to_string(),
        ))
            .get(filename)?;
        let mut file = std::fs::File::open(&model_path)?;
        let start = std::time::Instant::now();
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        let model = Qwen2::from_gguf(model, &mut file, &Device::Cpu)?;
        Ok(model)
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn main() -> anyhow::Result<()> {
    let config = ModelConfig::default();
    let mut p = Readline::default().prompt()?;

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        config.temperature, config.repeat_penalty, config.repeat_last_n
    );

    let mut model = config.model()?;

    let tokenizer = config.tokenizer()?;
    let mut tos = TokenOutputStream::new(tokenizer);
    println!(">> Hello, what can I do for you?");
    loop {
        match p.run() {
            Ok(input) => {
                if input.len() > 0 {
                    println!("\n>>> ");
                    std::io::stdout().flush()?;
                    tos = generate(&input.to_string(), &config, tos, &mut model)?;
                }
            }
            Err(_) => {
                println!("Bye!");
                break;
            }
        }
    }

    Ok(())
}

fn generate(prompt: &str, config: &ModelConfig, mut tos: TokenOutputStream, model: &mut Qwen2) -> anyhow::Result<TokenOutputStream> {
    let prompt_str = format!("<｜User｜>{prompt}<｜Assistant｜>");
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();

    let to_sample = config.sample_len.saturating_sub(1);
    let mut all_tokens = vec![];
    let mut logits_processor = {
        let temperature = config.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (config.top_k, config.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(config.seed, sampling)
    };
    let mut next_token = if !config.split_prompt {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos_token = "<｜end▁of▁sentence｜>";

    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if config.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                config.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if next_token == eos_token {
            break;
        };
    }
    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }
    println!("");
    std::io::stdout().flush()?;
    Ok(tos)
}
