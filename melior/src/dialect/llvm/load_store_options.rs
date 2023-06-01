use crate::{
    ir::{
        attribute::{ArrayAttribute, IntegerAttribute},
        Attribute, Identifier,
    },
    Context,
};

#[derive(Debug, Default)]
pub struct LoadStoreOptions<'c> {
    pub align: Option<IntegerAttribute<'c>>,
    pub volatile: bool,
    pub nontemporal: bool,
    pub access_groups: Option<ArrayAttribute<'c>>,
    pub alias_scopes: Option<ArrayAttribute<'c>>,
    pub noalias_scopes: Option<ArrayAttribute<'c>>,
    pub tbaa: Option<ArrayAttribute<'c>>,
}

impl<'c> LoadStoreOptions<'c> {
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
