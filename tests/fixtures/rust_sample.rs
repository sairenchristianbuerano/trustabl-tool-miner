use reqwest;

/// Fetches a remote document.
#[tool]
pub async fn fetch_doc(url: String) -> String {
    let body = reqwest::get(&url).await.unwrap().text().await.unwrap();
    body
}

#[tool]
fn run_shell(cmd: String) -> String {
    let out = std::process::Command::new("sh").arg("-c").arg(&cmd).output();
    format!("{:?}", out)
}
