mod error;
mod generation;
mod input;
mod operation;
mod r#trait;
mod r#type;
mod utility;

use self::{
    error::Error,
    generation::generate_operation,
    utility::{sanitize_documentation, sanitize_snake_case_identifier},
};
pub use input::DialectInput;
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::{env, fmt::Display, path::Path, process::Command, str};
use tblgen::{record::Record, record_keeper::RecordKeeper, TableGenParser};

const LLVM_MAJOR_VERSION: usize = 17;

pub fn generate_dialect(input: DialectInput) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut parser = TableGenParser::new();

    if let Some(source) = input.table_gen() {
        parser = parser.add_source(source).map_err(create_syn_error)?;
    }

    if let Some(file) = input.td_file() {
        parser = parser.add_source_file(file).map_err(create_syn_error)?;
    }

    // spell-checker: disable-next-line
    let llvm_include_directory = llvm_config("--includedir")?;

    for path in input
        .include_directories()
        .chain([llvm_include_directory.as_str()])
    {
        parser = parser.add_include_path(path);
    }

    let keeper = parser.parse().map_err(Error::Parse)?;

    let dialect = generate_dialect_module(
        input.name(),
        keeper
            .all_derived_definitions("Dialect")
            .find(|definition| definition.str_value("name") == Ok(input.name()))
            .ok_or_else(|| create_syn_error("dialect not found"))?,
        &keeper,
    )
    .map_err(|error| error.add_source_info(keeper.source_info()))?;

    Ok(quote! { #dialect }.into())
}

fn generate_dialect_module(
    name: &str,
    dialect: Record,
    record_keeper: &RecordKeeper,
) -> Result<proc_macro2::TokenStream, Error> {
    let dialect_name = dialect.name()?;
    let operations = record_keeper
        .all_derived_definitions("Op")
        .map(Operation::new)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(|operation| {
            Ok::<_, Error>(if operation.dialect_name()? == dialect_name {
                Some(generate_operation(operation)?)
            } else {
                None
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let doc = format!(
        "`{name}` dialect.\n\n{}",
        sanitize_documentation(dialect.str_value("description").unwrap_or(""),)?
    );
    let name = sanitize_snake_case_identifier(name)?;

    Ok(quote! {
        #[doc = #doc]
        pub mod #name {
            #(#operations)*
        }
    })
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prefix = env::var(format!("MLIR_SYS_{}0_PREFIX", LLVM_MAJOR_VERSION))
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let call = format!(
        "{} --link-static {}",
        prefix.join("llvm-config").display(),
        argument
    );

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}

fn create_syn_error(error: impl Display) -> syn::Error {
    syn::Error::new(Span::call_site(), format!("{}", error))
}
