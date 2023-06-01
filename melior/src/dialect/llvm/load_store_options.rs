use crate::{
    ir::{
        attribute::{ArrayAttribute, IntegerAttribute},
        Attribute, Identifier,
    },
    Context,
};

/// Load/store operation options.
#[derive(Debug, Default)]
pub struct LoadStoreOptions<'c> {
    align: Option<IntegerAttribute<'c>>,
    volatile: bool,
    nontemporal: bool,
    access_groups: Option<ArrayAttribute<'c>>,
    alias_scopes: Option<ArrayAttribute<'c>>,
    noalias_scopes: Option<ArrayAttribute<'c>>,
    tbaa: Option<ArrayAttribute<'c>>,
}

impl<'c> LoadStoreOptions<'c> {
    /// Creates load/store options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an alignment.
    pub fn align(mut self, align: IntegerAttribute<'c>) -> Self {
        self.align = Some(align);
        self
    }

    /// Sets a volatile flag.
    pub fn volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    /// Sets a nontemporal flag.
    pub fn nontemporal(mut self) -> Self {
        self.nontemporal = true;
        self
    }

    /// Sets access groups.
    pub fn access_groups(mut self, access_groups: ArrayAttribute<'c>) -> Self {
        self.access_groups = Some(access_groups);
        self
    }

    /// Sets alias scopes.
    pub fn alias_scopes(mut self, alias_scopes: ArrayAttribute<'c>) -> Self {
        self.alias_scopes = Some(alias_scopes);
        self
    }

    /// Sets noalias scopes.
    pub fn nonalias_scopes(mut self, noalias_scopes: ArrayAttribute<'c>) -> Self {
        self.noalias_scopes = Some(noalias_scopes);
        self
    }

    /// Sets tbaa.
    pub fn tbaa(mut self, tbaa: ArrayAttribute<'c>) -> Self {
        self.tbaa = Some(tbaa);
        self
    }

    pub(super) fn into_attributes(
        self,
        context: &'c Context,
    ) -> Vec<(Identifier<'c>, Attribute<'c>)> {
        let mut attributes = Vec::with_capacity(7);

        if let Some(align) = self.align {
            attributes.push((Identifier::new(context, "alignment"), align.into()));
        }

        if self.volatile {
            attributes.push((
                Identifier::new(context, "volatile_"),
                Attribute::unit(context),
            ));
        }

        if self.nontemporal {
            attributes.push((
                Identifier::new(context, "nontemporal"),
                Attribute::unit(context),
            ));
        }

        if let Some(alias_scopes) = self.alias_scopes {
            attributes.push((
                Identifier::new(context, "alias_scopes"),
                alias_scopes.into(),
            ));
        }

        if let Some(noalias_scopes) = self.noalias_scopes {
            attributes.push((
                Identifier::new(context, "noalias_scopes"),
                noalias_scopes.into(),
            ));
        }

        if let Some(tbaa) = self.tbaa {
            attributes.push((Identifier::new(context, "tbaa"), tbaa.into()));
        }

        attributes
    }
}
