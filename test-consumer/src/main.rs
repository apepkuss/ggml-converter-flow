use reqwest::Error;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    name: ModelType,
    quant_info: QuantInfo,
}

#[derive(Debug, Deserialize, Serialize)]
enum ModelType {
    /// meta-llama/Llama-2-7b-hf
    Llama2_7b,
    /// meta-llama/Llama-2-7b-chat-hf
    Llama2Chat7b,
    /// LinkSoul/Chinese-Llama-2-7b
    Llama2Chinese7b,
}

#[derive(Debug, Deserialize, Serialize)]
enum QuantInfo {
    /// q4_0
    Q4,
    /// q8_0
    Q8,
    /// f16
    F16,
    /// f32
    F32,
}

#[derive(Debug, Deserialize, Serialize)]
struct ConversionResult {
    download_url: String,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let model_info = ModelInfo {
        name: ModelType::Llama2_7b, // "meta-llama/Llama-2-7b-hf".to_string(),
        quant_info: QuantInfo::Q4,  // "q4_0".to_string(),
    };

    let client = reqwest::Client::new();

    let response = client
        .post("http://localhost:3000/json")
        .json(&model_info)
        .send()
        .await?;

    println!("{:?}", response);

    let conversion_result = response.json::<ConversionResult>().await?;
    println!("download url: {}", conversion_result.download_url);

    Ok(())
}
