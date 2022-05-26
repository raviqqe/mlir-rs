use std::error::Error;
use std::process::Command;
use std::str;

fn main() {
    run().unwrap();
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rustc-link-lib=stdc++");
    println!(
        "cargo:rustc-link-search=all={}",
        str::from_utf8(&Command::new("llvm-config").arg("--libdir").output()?.stdout)?
    );

    Ok(())
}
