macro_rules! from_raw_subtypes {
    ($type:ident,) => {};
    ($type:ident, $name:ident $(, $names:ident)* $(,)?) => {
        impl<'c> From<$name<'c>> for $type<'c> {
            fn from(value: $name<'c>) -> Self {
                unsafe { Self::from_raw(value.to_raw()) }
            }
        }

        from_raw_subtypes!($type, $($names,)*);
    };
}
