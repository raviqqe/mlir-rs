use proc_macro2::Ident;
use quote::format_ident;
use std::ops::Deref;
use syn::{bracketed, parse::Parse, punctuated::Punctuated, LitStr, Token};

pub struct DialectInput {
    name: String,
    table_gen: Option<String>,
    td_file: Option<String>,
    includes: Vec<String>,
}

impl DialectInput {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn table_gen(&self) -> Option<&str> {
        self.table_gen.as_deref()
    }

    pub fn td_file(&self) -> Option<&str> {
        self.td_file.as_deref()
    }

    pub fn includes(&self) -> impl Iterator<Item = &str> {
        self.includes.iter().map(Deref::deref)
    }
}

impl Parse for DialectInput {
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
            table_gen: tablegen,
            td_file,
            includes,
        })
    }
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
