macro_rules! attribute_traits {
    ($name: ident) => {
        impl<'c> Display for FloatAttribute<'c> {
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> Debug for FloatAttribute<'c> {
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                Display::fmt(self, formatter)
            }
        }
    };
}
