mod error;
mod operation;
mod types;
mod utility;

use self::{
    error::Error,
    utility::{sanitize_documentation, sanitize_snake_case_name},
};
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use std::{env, fmt::Display, path::Path, process::Command, str};
use syn::{bracketed, parse::Parse, punctuated::Punctuated, LitStr, Token};
use tblgen::{record::Record, record_keeper::RecordKeeper, TableGenParser};

const LLVM_MAJOR_VERSION: usize = 16;

fn dialect_module(
    name: &str,
    dialect: Record,
    record_keeper: &RecordKeeper,
) -> Result<proc_macro2::TokenStream, Error> {
    let operations = record_keeper
        .all_derived_definitions("Op")
        .map(Operation::from_def)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|operation| operation.dialect.name() == dialect.name())
        .collect::<Vec<_>>();

    let doc = format!(
        "`{name}` dialect.\n\n{}",
        sanitize_documentation(&unindent::unindent(
            dialect.str_value("description").unwrap_or(""),
        ))?
    );
    let name = sanitize_snake_case_name(name);

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
        let ident = input.parse::<Ident>()?;

        input.parse::<Token![:]>()?;

        if ident == format_ident!("name") {
            Ok(Self::Name(input.parse()?))
        } else if ident == format_ident!("tablegen") {
            Ok(Self::TableGen(input.parse()?))
        } else if ident == format_ident!("td_file") {
            Ok(Self::TdFile(input.parse()?))
        } else if ident == format_ident!("include_dirs") {
            let content;
            bracketed!(content in input);
            Ok(Self::Includes(
                Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?,
            ))
        } else {
            Err(input.error(format!("invalid field {}", ident)))
        }
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
        let mut name = None;
        let mut tablegen = None;
        let mut td_file = None;
        let mut includes = vec![];

        for item in Punctuated::<InputField, Token![,]>::parse_terminated(input)? {
            match item {
                InputField::Name(field) => name = Some(field.value()),
                InputField::TableGen(td) => tablegen = Some(td.value()),
                InputField::TdFile(file) => td_file = Some(file.value()),
                InputField::Includes(field) => {
                    includes = field.into_iter().map(|literal| literal.value()).collect()
                }
            }
        }

        Ok(Self {
            name: name.ok_or(input.error("dialect name required"))?,
            tablegen,
            td_file,
            includes,
        })
    }
}

pub fn generate_dialect(
    input: DialectMacroInput,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut td_parser = TableGenParser::new();

    if let Some(source) = &input.tablegen {
        td_parser = td_parser.add_source(source).map_err(create_syn_error)?;
    }

    if let Some(file) = &input.td_file {
        td_parser = td_parser.add_source_file(file).map_err(create_syn_error)?;
    }

    // spell-checker: disable-next-line
    for include in input.includes.iter().chain([&llvm_config("--includedir")?]) {
        td_parser = td_parser.add_include_path(include);
    }

    let keeper = td_parser.parse().map_err(Error::Parse)?;

    let dialect = dialect_module(
        &input.name,
        keeper
            .all_derived_definitions("Dialect")
            .find(|def| def.str_value("name") == Ok(&input.name))
            .ok_or_else(|| create_syn_error("dialect not found"))?,
        &keeper,
    )
    .map_err(|error| error.add_source_info(keeper.source_info()))?;

    Ok(quote! { #dialect }.into())
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
