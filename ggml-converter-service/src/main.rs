use axum::{
    body::{self, Body},
    extract::Query,
    http::header::{HeaderMap, HeaderName, HeaderValue},
    response::{Headers, Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use http::{StatusCode, Uri};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Command;
use std::{collections::HashMap, sync::Mutex, time::Instant};

use once_cell::sync::Lazy;

static MODELS: Lazy<Mutex<HashMap<String, String>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        String::from("meta-llama/Llama-2-7b-hf"),
        String::from("https://huggingface.co/meta-llama/Llama-2-7b-hf"),
    );
    map.insert(
        String::from("meta-llama/Llama-2-7b-chat-hf"),
        String::from("https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"),
    );
    Mutex::new(map)
});

// We've already seen returning &'static str
async fn plain_text() -> &'static str {
    "foo"
}

// String works too and will get a `text/plain; charset=utf-8` content-type
async fn plain_text_string(uri: Uri) -> String {
    format!("Hi from {}", uri.path())
}

// Bytes will get a `application/octet-stream` content-type
async fn bytes() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

// `()` gives an empty response
async fn empty() {}

// `StatusCode` gives an empty response with that status code
async fn empty_with_status() -> StatusCode {
    StatusCode::NOT_FOUND
}

// A tuple of `StatusCode` and something that implements `IntoResponse` can
// be used to override the status code
async fn with_status() -> (StatusCode, &'static str) {
    (StatusCode::INTERNAL_SERVER_ERROR, "Something went wrong")
}

// A tuple of `HeaderMap` and something that implements `IntoResponse` can
// be used to override the headers
async fn with_headers() -> (HeaderMap, &'static str) {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-foo"),
        HeaderValue::from_static("foo"),
    );
    (headers, "foo")
}

// You can also override both status and headers at the same time
async fn with_headers_and_status() -> (StatusCode, HeaderMap, &'static str) {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-foo"),
        HeaderValue::from_static("foo"),
    );
    (StatusCode::INTERNAL_SERVER_ERROR, headers, "foo")
}

// `Headers` makes building the header map easier and `impl Trait` is easier
// so you don't have to write the whole type
async fn with_easy_headers() -> impl IntoResponse {
    Headers(vec![("x-foo", "foo")])
}

// `Html` gives a content-type of `text/html`
async fn html() -> Html<&'static str> {
    Html("<h1>Hello, World!</h1>")
}

// `Json` gives a content-type of `application/json` and works with any type
// that implements `serde::Serialize`
async fn json() -> Json<Value> {
    Json(json!({ "data": 42 }))
}

// `Result<T, E>` where `T` and `E` implement `IntoResponse` is useful for
// returning errors
async fn result() -> Result<&'static str, StatusCode> {
    Ok("all good")
}

// `Response` gives full control
async fn response() -> Response<Body> {
    Response::builder().body(Body::empty()).unwrap()
}

//eg: query?a=1&b=1.0&c=xxx
async fn query(Query(params): Query<HashMap<String, String>>) -> String {
    for (key, value) in &params {
        println!("key:{},value:{}", key, value);
    }
    format!("{:?}", params)
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    name: ModelType,
    quant_info: QuantInfo,
}

#[derive(Debug, Deserialize, Serialize)]
enum ModelType {
    Llama2_7b,
    Llama2Chat7b,
    Llama2Chinese7b,
}
impl From<ModelType> for String {
    fn from(model_type: ModelType) -> Self {
        match model_type {
            ModelType::Llama2_7b => "meta-llama/Llama-2-7b-hf".to_string(),
            ModelType::Llama2Chat7b => "meta-llama/Llama-2-7b-chat-hf".to_string(),
            ModelType::Llama2Chinese7b => "LinkSoul/Chinese-Llama-2-7b".to_string(),
        }
    }
}
impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_type = match self {
            ModelType::Llama2_7b => "meta-llama/Llama-2-7b-hf",
            ModelType::Llama2Chat7b => "meta-llama/Llama-2-7b-chat-hf",
            ModelType::Llama2Chinese7b => "LinkSoul/Chinese-Llama-2-7b",
        };
        write!(f, "{}", model_type)
    }
}

#[derive(Debug, Deserialize, Serialize)]
enum QuantInfo {
    Q4,
    Q8,
    F16,
    F32,
}
impl std::fmt::Display for QuantInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let quant_info = match self {
            QuantInfo::Q4 => "q4_0",
            QuantInfo::Q8 => "q8_0",
            QuantInfo::F16 => "f16",
            QuantInfo::F32 => "f32",
        };
        write!(f, "{}", quant_info)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ConversionResult {
    download_url: String,
}

// json request
async fn json_request(Json(model_info): Json<ModelInfo>) -> Json<ConversionResult> {
    println!("{:?}", &model_info);

    // download and build llama.cpp
    let llama_cpp_dir = download_and_build_llama_cpp().await.unwrap();
    dbg!(&llama_cpp_dir);

    // download llama2 models
    let model_repo_dir = download_llama2_models(&model_info).await.unwrap();
    dbg!(&model_repo_dir);

    // convert the target model to ggml
    let curr_dir = std::env::current_dir().unwrap();
    let root_dir = curr_dir.parent().unwrap();
    let outputs_dir = root_dir.join("outputs");
    if !outputs_dir.exists() {
        std::fs::create_dir(outputs_dir.as_path()).unwrap();
    }
    let out_filename = format!(
        "{}-ggml.{}",
        model_info
            .name
            .to_string()
            .split('/')
            .collect::<Vec<&str>>()[1],
        "bin"
    );
    let outfile = outputs_dir.join(out_filename.as_str());
    convert_to_ggml(
        llama_cpp_dir.as_path(),
        model_repo_dir.as_path(),
        outfile.as_path(),
    )
    .await
    .unwrap();

    // quantize the ggml model
    let quantized_filename = format!(
        "{}-ggml-{}.{}",
        model_info
            .name
            .to_string()
            .split('/')
            .collect::<Vec<&str>>()[1],
        model_info.quant_info,
        "bin"
    );
    let quantized_outfile = outputs_dir.join(quantized_filename.as_str());
    quantize_ggml(
        llama_cpp_dir.as_path(),
        outfile.as_path(),
        model_info.quant_info,
        quantized_outfile.as_path(),
    )
    .await
    .unwrap();

    println!("Done.");

    let res = ConversionResult {
        download_url: quantized_outfile.to_str().unwrap().to_string(),
    };

    Json(res)
}

// From https://github.com/ggerganov/llama.cpp/tags
const CODE_BASE: &str = "d2a4366";

async fn download_and_build_llama_cpp() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let curr_dir = std::env::current_dir()?;
    let llama_cpp_dir = curr_dir.parent().unwrap().join("llama.cpp");

    // download
    if !llama_cpp_dir.exists() {
        let url = format!(
            "https://github.com/ggerganov/llama.cpp/archive/refs/tags/master-{CODE_BASE}.tar.gz"
        );

        let status = Command::new("wget").arg(&url).status()?;
        println!("status: {:?}", status);

        let status = Command::new("tar")
            .arg("-zxvf")
            .arg("master-d2a4366.tar.gz")
            .status();
        println!("status: {:?}", status);

        let status = Command::new("rm")
            .arg("-rf")
            .arg(format!("master-{CODE_BASE}.tar.gz").as_str())
            .status();
        println!("status: {:?}", status);

        let status = Command::new("mv")
            .arg(format!("llama.cpp-master-{CODE_BASE}").as_str())
            .arg("llama.cpp")
            .status();
        println!("status: {:?}", status);

        if !std::path::Path::new("llama.cpp").exists() {
            panic!("Not found llama.cpp directory");
        }
    } else {
        println!("llama.cpp directory already exists");
    }

    // build
    let quantizer = llama_cpp_dir.join("quantize");
    if quantizer.exists() && quantizer.is_file() {
        println!("Already build llama.cpp");
    } else {
        std::env::set_current_dir(llama_cpp_dir.as_path())?;

        // build llama.cpp
        let status = Command::new("make").arg("-j").status();
        println!("status: {:?}", status);

        // check if the build process is successful
        let status = Command::new("./quantize").arg("--help").status()?;
        println!("status: {:?}", status);

        std::env::set_current_dir(curr_dir.as_path())?;
    }

    Ok(llama_cpp_dir)
}

async fn download_llama2_models(
    model_info: &ModelInfo,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut success = false;
    let mut retries = 0;

    let curr_dir = std::env::current_dir()?;
    let models_dir = curr_dir.parent().unwrap().join("models");
    if !models_dir.exists() {
        std::fs::create_dir(models_dir.as_path())?;
    }

    let model_repo_dir = models_dir.join(
        model_info
            .name
            .to_string()
            .split('/')
            .collect::<Vec<&str>>()[1],
    );
    if model_repo_dir.exists() {
        println!("Model '{}' already exists", model_info.name);
    } else {
        let locked = MODELS.lock().unwrap();
        let url = locked
            .get(model_info.name.to_string().as_str())
            .ok_or(format!(
                "Failed to get the url of the model '{}'",
                model_info.name.to_string()
            ))?;

        println!("Downloading from {url}...");

        while !success && retries < 3 {
            println!("({retries}) Git clone llama2 models...");

            let output = Command::new("git").arg("clone").arg(url).output();

            match output {
                Ok(output) if output.status.success() => {
                    success = true;
                    println!("Git clone succeeded!");
                }
                _ => {
                    retries += 1;
                    println!("output: {:?}", output);
                    println!("Git clone failed, retry again...");
                }
            }
        }

        if !success {
            println!("Git clone failed after 3 retries.");
        }
    }

    Ok(model_repo_dir)
}

async fn convert_to_ggml(
    llama_cpp_dir: &std::path::Path,
    model_repo_dir: &std::path::Path,
    outfile: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let converter = llama_cpp_dir.join("convert.py");
    println!("converter: {:?}", converter.as_path());

    println!("out_file: {:?}", outfile);
    if outfile.exists() {
        std::fs::remove_file(outfile)?;
    }

    if converter.exists() && converter.is_file() {
        println!(
            "================ Start to convert {} to ggml...",
            model_repo_dir.file_name().unwrap().to_str().unwrap()
        );

        let start = Instant::now();
        let output = Command::new("python3")
            .arg(converter)
            .arg(model_repo_dir)
            .arg("--outfile")
            .arg(outfile)
            .output()?;
        let elapsed = Instant::now() - start;

        match output.status.success() {
            true => println!("The conversion took {:?} seconds.", elapsed.as_secs()),
            false => println!("Conversion failed!"),
        }
    } else {
        panic!("Not found converter.py");
    }

    Ok(())
}

/// Quantize the ggml model
async fn quantize_ggml(
    llama_cpp_dir: &std::path::Path,
    model: &std::path::Path,
    quant_info: QuantInfo,
    outfile: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let quantizer = llama_cpp_dir.join("quantize");
    println!("quantizer: {:?}", quantizer.as_path());

    if outfile.exists() {
        std::fs::remove_file(outfile)?;
    }

    // quantize
    if quantizer.exists() && quantizer.is_file() {
        println!(
            "============== Start to quantize {} ...",
            model.file_name().unwrap().to_str().unwrap()
        );

        let start = Instant::now();
        let output = Command::new(quantizer.as_os_str())
            .arg(model)
            .arg(outfile)
            .arg(quant_info.to_string())
            .output()?;
        let elapsed = Instant::now() - start;

        match output.status.success() {
            true => println!("The quantization took {:?} seconds.", elapsed.as_secs()),
            false => println!("Quantization failed!"),
        }
    } else {
        panic!("Not found quantizer");
    }

    Ok(())
}

#[derive(Serialize)]
struct Blog {
    title: String,
    author: String,
    summary: String,
}

async fn blog_struct() -> Json<Blog> {
    let blog = Blog {
        title: "axum笔记(2)-response".to_string(),
        author: "菩提树下的杨过".to_string(),
        summary: "response各种示例".to_string(),
    };
    Json(blog)
}

async fn blog_struct_cn() -> (HeaderMap, Json<Blog>) {
    let blog = Blog {
        title: "axum笔记(2)-response".to_string(),
        author: "菩提树下的杨过".to_string(),
        summary: "response各种示例".to_string(),
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json;charset=utf-8"),
    );
    (headers, Json(blog))
}

struct CustomError {
    msg: String,
}

impl IntoResponse for CustomError {
    fn into_response(self) -> Response {
        let body = body::boxed(body::Full::from(self.msg));
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(body)
            .unwrap()
    }
}

async fn custom_error() -> Result<&'static str, CustomError> {
    Err(CustomError {
        msg: "Opps!".to_string(),
    })
}

#[tokio::main]
async fn main() {
    println!("Service started on port 3000");

    // our router
    let app = Router::new()
        .route("/plain_text", get(plain_text))
        .route("/plain_text_string", get(plain_text_string))
        .route("/bytes", get(bytes))
        .route("/empty", get(empty))
        .route("/empty_with_status", get(empty_with_status))
        .route("/with_status", get(with_status))
        .route("/with_headers", get(with_headers))
        .route("/with_headers_and_status", get(with_headers_and_status))
        .route("/with_easy_headers", get(with_easy_headers))
        .route("/html", get(html))
        .route("/json", get(json))
        .route("/result", get(result))
        .route("/response", get(response))
        .route("/blog", get(blog_struct))
        .route("/blog_cn", get(blog_struct_cn))
        .route("/custom_error", get(custom_error))
        .route("/query", get(query))
        .route("/json", post(json_request));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
