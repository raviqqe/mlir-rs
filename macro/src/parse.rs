use proc_macro2::Ident;
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    LitStr, Result, Token,
};

pub struct IdentifierList {
    identifiers: Vec<Ident>,
}

impl IdentifierList {
    pub fn identifiers(&self) -> &[Ident] {
        &self.identifiers
    }
}

impl Parse for IdentifierList {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            identifiers: Punctuated::<Ident, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        })
    }
}

pub struct DialectOperationSet {
    dialect: Ident,
    identifiers: IdentifierList,
}

impl DialectOperationSet {
    pub const fn dialect(&self) -> &Ident {
        &self.dialect
    }

    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}

impl Parse for DialectOperationSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let dialect = Ident::parse(input)?;
        <Token![,]>::parse(input)?;

        Ok(Self {
            dialect,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}

pub struct PassSet {
    prefix: LitStr,
    identifiers: IdentifierList,
}

impl PassSet {
    pub const fn prefix(&self) -> &LitStr {
        &self.prefix
    }

    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}

impl Parse for PassSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let prefix = input.parse()?;
        <Token![,]>::parse(input)?;

        Ok(Self {
            prefix,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}
