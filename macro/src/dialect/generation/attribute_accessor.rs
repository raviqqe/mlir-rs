use crate::dialect::{
    error::Error,
    operation::{Attribute, OperationFieldLike},
    utility::sanitize_snake_case_identifier,
};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_attribute_accessors(attribute: &Attribute) -> Result<TokenStream, Error> {
    let getter = generate_getter(attribute)?;
    let setter = generate_setter(attribute)?;
    let remover = generate_remover(attribute)?;

    Ok(quote! {
        #getter
        #setter
        #remover
    })
}

fn generate_getter(attribute: &Attribute) -> Result<TokenStream, Error> {
    let name = attribute.name();

    let identifier = attribute.singular_identifier();
    let return_type = attribute.return_type();
    let body = if attribute.is_unit() {
        quote! { self.operation.attribute(#name).is_some() }
    } else {
        // TODO Handle returning `melior::Attribute`.
        quote! { Ok(self.operation.attribute(#name)?.try_into()?) }
    };

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}

fn generate_setter(attribute: &Attribute) -> Result<TokenStream, Error> {
    let name = attribute.name();

    let body = if attribute.is_unit() {
        quote! {
            if value {
                self.operation.set_attribute(#name, Attribute::unit(&self.operation.context()));
            } else {
                self.operation.remove_attribute(#name)
            }
        }
    } else {
        quote! {
            self.operation.set_attribute(#name, &value.into());
        }
    };

    let ident = sanitize_snake_case_identifier(&format!("set_{}", attribute.name()))?;
    let r#type = attribute.parameter_type();

    Ok(quote! {
        pub fn #ident(&mut self, context: &'c ::melior::Context, value: #r#type) {
            #body
        }
    })
}

fn generate_remover(attribute: &Attribute) -> Result<Option<TokenStream>, Error> {
    Ok(if attribute.is_unit() || attribute.is_optional() {
        let name = attribute.name();
        let ident = sanitize_snake_case_identifier(&format!("remove_{}", attribute.name()))?;

        Some(quote! {
            pub fn #ident(&mut self, context: &'c ::melior::Context) -> Result<(), ::melior::Error> {
                self.operation.remove_attribute(#name)
            }
        })
    } else {
        None
    })
}
