use super::{FieldKind, OperationField, SequenceInfo, VariadicKind};
use crate::utility::sanitize_name_snake;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

impl<'a> OperationField<'a> {
    fn getter_impl(&self) -> Option<TokenStream> {
        match &self.kind {
            FieldKind::Operand(constraint) | FieldKind::Result(constraint) => {
                let kind = self.kind.as_str();
                let kind_ident = format_ident!("{}", kind);
                let plural = format_ident!("{}s", kind);
                let count = format_ident!("{}_count", kind);
                let SequenceInfo { index, len } = self
                    .seq_info
                    .as_ref()
                    .expect("operands and results need sequence info");
                let variadic_kind = self
                    .variadic_info
                    .as_ref()
                    .expect("operands and results need variadic info");
                Some(match variadic_kind {
                    VariadicKind::Simple {
                        seen_variable_length,
                    } => {
                        // At most one variable length group
                        if constraint.is_variable_length() {
                            if constraint.is_optional() {
                                // Optional element, and some singular elements.
                                // Only present if the amount of groups is at least the number of
                                // elements.
                                quote! {
                                  if self.operation.#count() < #len {
                                    None
                                  } else {
                                    self.operation.#kind_ident(#index).ok()
                                  }
                                }
                            } else {
                                // A variable length group
                                // Length computed by subtracting the amount of other
                                // singular elements from the number of elements.
                                quote! {
                                  let group_length = self.operation.#count() - #len + 1;
                                  self.operation.#plural().skip(#index).take(group_length)
                                }
                            }
                        } else if *seen_variable_length {
                            // Single element after variable length group
                            // Compute the length of that variable group and take the next element
                            let error = format!("operation should have this {}", kind);
                            quote! {
                                let group_length = self.operation.#count() - #len + 1;
                                self.operation.#kind_ident(#index + group_length - 1).expect(#error)
                            }
                        } else {
                            // All elements so far are singular
                            let error = format!("operation should have this {}", kind);
                            quote! {
                                self.operation.#kind_ident(#index).expect(#error)
                            }
                        }
                    }
                    VariadicKind::SameSize {
                        num_variable_length,
                        num_preceding_simple,
                        num_preceding_variadic,
                    } => {
                        let error = format!("operation should have this {}", kind);
                        let compute_start_length = quote! {
                            let total_var_len = self.operation.#count() - #num_variable_length + 1;
                            let group_len = total_var_len / #num_variable_length;
                            let start = #num_preceding_simple + #num_preceding_variadic * group_len;
                        };

                        let get_elements = if constraint.is_variable_length() {
                            quote! {
                                self.operation.#plural().skip(start).take(group_len)
                            }
                        } else {
                            quote! {
                                self.operation.#kind_ident(start).expect(#error)
                            }
                        };
                        quote! { #compute_start_length #get_elements }
                    }
                    VariadicKind::AttrSized {} => {
                        let error = format!("operation should have this {}", kind);
                        let attr_name = format!("{}_segment_sizes", kind);
                        let attr_missing_error = format!("operation has {} attribute", attr_name);
                        let compute_start_length = quote! {
                            let attribute =
                                ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
                                    self.operation
                                        .attribute(#attr_name)
                                        .expect(#attr_missing_error)
                                ).expect("is a DenseI32ArrayAttribute");
                            let start = (0..#index)
                                            .map(|i| attribute.element(i)
                                            .expect("has segment size"))
                                            .sum::<i32>() as usize;
                            let group_len = attribute
                                                .element(#index)
                                                .expect("has segment size") as usize;
                        };
                        let get_elements = if !constraint.is_variable_length() {
                            quote! {
                                self.operation.#kind_ident(start).expect(#error)
                            }
                        } else if constraint.is_optional() {
                            quote! {
                                if group_len == 0 {
                                    None
                                } else {
                                    self.operation.#kind_ident(start).ok()
                                }
                            }
                        } else {
                            quote! {
                                self.operation.#plural().skip(start).take(group_len)
                            }
                        };
                        quote! { #compute_start_length #get_elements }
                    }
                })
            }
            FieldKind::Successor(s) => {
                let SequenceInfo { index, .. } = self
                    .seq_info
                    .as_ref()
                    .expect("successors need sequence info");
                Some(if s.is_variadic() {
                    // Only the last successor can be variadic
                    quote! {
                        self.operation.successors().skip(#index)
                    }
                } else {
                    quote! {
                        self.operation.successor(#index).expect("operation should have this successor")
                    }
                })
            }
            FieldKind::Region(r) => {
                let SequenceInfo { index, .. } =
                    self.seq_info.as_ref().expect("regions need sequence info");
                Some(if r.is_variadic() {
                    // Only the last region can be variadic
                    quote! {
                        self.operation.regions().skip(#index)
                    }
                } else {
                    quote! {
                        self.operation.region(#index).expect("operation should have this region")
                    }
                })
            }
            FieldKind::Attribute(a) => {
                let n = &self.name;
                let attr_error = format!("operation should have attribute {}", n);
                let type_error = format!("{} should be a {}", n, a.storage_type());
                Some(if a.is_unit() {
                    quote! { self.operation.attribute(#n).is_some() }
                } else if a.is_optional() {
                    quote! {
                        self.operation
                            .attribute(#n)
                            .map(|a| a.try_into().expect(#type_error))
                    }
                } else {
                    quote! {
                        self.operation
                            .attribute(#n)
                            .expect(#attr_error)
                            .try_into()
                            .expect(#type_error)
                    }
                })
            }
        }
    }

    fn remover_impl(&self) -> Option<TokenStream> {
        match &self.kind {
            FieldKind::Attribute(a) => {
                let n = &self.name;

                if a.is_unit() || a.is_optional() {
                    Some(quote! {
                      let _ = self.operation.remove_attribute(#n);
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn setter_impl(&self) -> Option<TokenStream> {
        match &self.kind {
            FieldKind::Attribute(a) => {
                let n = &self.name;

                Some(if a.is_unit() {
                    quote! {
                        if value {
                          self.operation.set_attribute(#n, Attribute::unit(&self.operation.context()));
                        } else {
                          let _ = self.operation.remove_attribute(#n);
                        }
                    }
                } else {
                    quote! {
                        self.operation.set_attribute(#n, &value.into());
                    }
                })
            }
            _ => None,
        }
    }

    pub fn accessors(&self) -> TokenStream {
        let setter = {
            let set_fn_ident = sanitize_name_snake(&format!("set_{}", self.name));
            self.setter_impl().map_or(quote!(), |imp| {
                let param_type = &self.param_type;
                quote! {
                    pub fn #set_fn_ident(&mut self, value: #param_type) {
                        #imp
                    }
                }
            })
        };
        let remover = {
            let remove_fn_ident = sanitize_name_snake(&format!("remove_{}", self.name));
            self.remover_impl().map_or(quote!(), |imp| {
                quote! {
                    pub fn #remove_fn_ident(&mut self) {
                        #imp
                    }
                }
            })
        };
        let getter = {
            let get_fn_ident = &self.sanitized;
            let return_type = &self.return_type;
            self.getter_impl().map_or(quote!(), |imp| {
                quote! {
                    pub fn #get_fn_ident(&self) -> #return_type {
                        #imp
                    }
                }
            })
        };
        quote! {
            #getter
            #setter
            #remover
        }
    }
}
