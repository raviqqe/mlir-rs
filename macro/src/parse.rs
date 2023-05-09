use proc_macro2::Ident;
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Result, Token,
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
    pub fn dialect(&self) -> &Ident {
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
