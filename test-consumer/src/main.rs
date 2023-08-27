use reqwest::Error;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    name: String,
    quant_info: String,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let model_info = ModelInfo {
        name: "meta-llama/Llama-2-7b-hf".to_string(),
        quant_info: "q4_0".to_string(),
    };

    let client = reqwest::Client::new();

    let res = client
        .post("http://localhost:3000/json")
        .json(&model_info)
        .send()
        .await?;

    println!("{:?}", res);

    Ok(())
}
