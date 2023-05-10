macro_rules! attribute_traits {
    ($name: ident) => {
        impl<'c> crate::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, formatter)
            }
        }
    };
}
