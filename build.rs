fn main() {
    println!("cargo::rustc-check-cfg=cfg(striped)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "x86_64" || arch == "aarch64" {
        println!("cargo::rustc-cfg=striped");
    }
}
