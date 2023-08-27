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
use std::collections::HashMap;
use std::process::Command;

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
    name: String,
    quant_info: String,
}

// json request
async fn json_request(Json(model): Json<ModelInfo>) -> String {
    println!("{:?}", model);

    // download ggml
    download_ggml().await.unwrap();

    "download_url".to_string()
}

// From https://github.com/ggerganov/llama.cpp/tags
const CODE_BASE: &str = "d2a4366";

async fn download_ggml() -> Result<(), Box<dyn std::error::Error>> {
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

    if std::path::Path::new("llama.cpp").exists() {
        let curr_dir = std::env::current_dir()?;
        let llama_cpp_dir = curr_dir.join("llama.cpp");
        std::env::set_current_dir(llama_cpp_dir)?;
        let status = Command::new("make").arg("-j").status();
        println!("status: {:?}", status);

        let status = Command::new("./quantize").arg("--help").status()?;
        println!("status: {:?}", status);
    } else {
        println!("Not found llama.cpp");
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
