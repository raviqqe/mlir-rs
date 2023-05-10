macro_rules! attribute_traits {
    ($name: ident) => {
        use std::{
            convert::TryFrom,
            fmt::{self, Debug, Display, Formatter},
        };

        impl<'c> AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> Display for $name<'c> {
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> Debug for $name<'c> {
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                Display::fmt(self, formatter)
            }
        }
    };
}
