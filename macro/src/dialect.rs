extern crate proc_macro;

mod error;
mod operation;
mod types;

use crate::utility::sanitize_name_snake;
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use std::{env, error::Error, fs::OpenOptions, io::Write, path::Path, process::Command, str};
use syn::{bracketed, parse::Parse, punctuated::Punctuated, LitStr, Token};
use tblgen::{record::Record, record_keeper::RecordKeeper, TableGenParser};

const LLVM_MAJOR_VERSION: usize = 16;

fn dialect_module<'a>(
    name: &str,
    dialect: Record<'a>,
    record_keeper: &'a RecordKeeper,
) -> Result<proc_macro2::TokenStream, error::Error> {
    let operations = record_keeper
        .all_derived_definitions("Op")
        .map(Operation::from_def)
        .filter_map(|operation: Result<Operation, _>| match operation {
            Ok(operation) => (operation.dialect.name() == dialect.name()).then_some(Ok(operation)),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut doc = format!("`{}` dialect.\n\n", name);
    doc.push_str(&unindent::unindent(
        dialect.str_value("description").unwrap_or(""),
    ));
    let name = sanitize_name_snake(name);
    Ok(quote! {
        #[doc = #doc]
        pub mod #name {
            #(#operations)*
        }
    })
}

enum InputField {
    Name(LitStr),
    TableGen(LitStr),
    TdFile(LitStr),
    Includes(Punctuated<LitStr, Token![,]>),
}

impl Parse for InputField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        let _: Token![:] = input.parse()?;
        if ident == format_ident!("name") {
            return Ok(Self::Name(input.parse()?));
        }
        if ident == format_ident!("tablegen") {
            return Ok(Self::TableGen(input.parse()?));
        }
        if ident == format_ident!("td_file") {
            return Ok(Self::TdFile(input.parse()?));
        }
        if ident == format_ident!("include_dirs") {
            let content;
            bracketed!(content in input);
            return Ok(Self::Includes(
                Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?,
            ));
        }

        Err(input.error(format!("invalid field {}", ident)))
    }
}

pub struct DialectMacroInput {
    name: String,
    tablegen: Option<String>,
    td_file: Option<String>,
    includes: Vec<String>,
}

impl Parse for DialectMacroInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let list = Punctuated::<InputField, Token![,]>::parse_terminated(input)?;
        let mut name = None;
        let mut tablegen = None;
        let mut td_file = None;
        let mut includes = None;

        for item in list {
            match item {
                InputField::Name(field) => name = Some(field.value()),
                InputField::TableGen(td) => tablegen = Some(td.value()),
                InputField::TdFile(file) => td_file = Some(file.value()),
                InputField::Includes(field) => {
                    includes = Some(field.into_iter().map(|literal| literal.value()).collect())
                }
            }
        }

        Ok(Self {
            name: name.ok_or(input.error("dialect name required"))?,
            tablegen,
            td_file,
            includes: includes.unwrap_or(Vec::new()),
        })
    }
}

// Writes `tablegen_compile_commands.yaml` for any TableGen file that is being
// parsed. See: https://mlir.llvm.org/docs/Tools/MLIRLSP/#tablegen-lsp-language-server--tblgen-lsp-server
fn emit_tablegen_compile_commands(td_file: &str, includes: &[String]) {
    let pwd = env::current_dir();
    if let Ok(pwd) = pwd {
        let path = pwd.join(td_file);
        let file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(pwd.join("tablegen_compile_commands.yml"));
        if let Ok(mut file) = file {
            writeln!(file, "--- !FileInfo:").unwrap();
            writeln!(file, "  filepath: \"{}\"", path.to_str().unwrap()).unwrap();
            let _ = writeln!(
                file,
                "  includes: \"{}\"",
                includes
                    .iter()
                    .map(|string| pwd.join(string).to_str().unwrap().to_owned())
                    .collect::<Vec<_>>()
                    .join(";")
            );
        }
    }
}

pub fn generate_dialect(mut input: DialectMacroInput) -> Result<TokenStream, Box<dyn Error>> {
    // spell-checker: disable-next-line
    input.includes.push(llvm_config("--includedir").unwrap());

    let mut td_parser = TableGenParser::new();

    if let Some(source) = &input.tablegen {
        td_parser = td_parser
            .add_source(source)
            .map_err(|e| syn::Error::new(Span::call_site(), format!("{}", e)))?;
    }
    if let Some(file) = &input.td_file {
        td_parser = td_parser
            .add_source_file(file)
            .map_err(|e| syn::Error::new(Span::call_site(), format!("{}", e)))?;
    }
    for include in &input.includes {
        td_parser = td_parser.add_include_path(include);
    }

    // spell-checker: disable-next-line
    if env::var("DIALECTGEN_TABLEGEN_COMPILE_COMMANDS").is_ok() {
        if let Some(td_file) = &input.td_file {
            emit_tablegen_compile_commands(td_file, &input.includes);
        }
    }

    let keeper = td_parser.parse().map_err(|_| error::Error::ParseError)?;

    let dialect_def = keeper
        .all_derived_definitions("Dialect")
        .find_map(|def| {
            def.str_value("name")
                .ok()
                .and_then(|n| if n == input.name { Some(def) } else { None })
        })
        .ok_or_else(|| syn::Error::new(Span::call_site(), "dialect not found"))?;
    let dialect = dialect_module(&input.name, dialect_def, &keeper)
        .map_err(|e| e.add_source_info(keeper.source_info()))?;

    Ok(quote! {
        #dialect
    }
    .into())
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
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
