(function() {var implementors = {
"bit_set":[["impl&lt;B: <a class=\"trait\" href=\"bit_vec/trait.BitBlock.html\" title=\"trait bit_vec::BitBlock\">BitBlock</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.usize.html\">usize</a>&gt; for <a class=\"struct\" href=\"bit_set/struct.BitSet.html\" title=\"struct bit_set::BitSet\">BitSet</a>&lt;B&gt;"]],
"bit_vec":[["impl&lt;B: <a class=\"trait\" href=\"bit_vec/trait.BitBlock.html\" title=\"trait bit_vec::BitBlock\">BitBlock</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.bool.html\">bool</a>&gt; for <a class=\"struct\" href=\"bit_vec/struct.BitVec.html\" title=\"struct bit_vec::BitVec\">BitVec</a>&lt;B&gt;"]],
"crossbeam_deque":[["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"enum\" href=\"crossbeam_deque/enum.Steal.html\" title=\"enum crossbeam_deque::Steal\">Steal</a>&lt;T&gt;&gt; for <a class=\"enum\" href=\"crossbeam_deque/enum.Steal.html\" title=\"enum crossbeam_deque::Steal\">Steal</a>&lt;T&gt;"]],
"dashmap":[["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>, S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;K&gt; for <a class=\"struct\" href=\"dashmap/struct.DashSet.html\" title=\"struct dashmap::DashSet\">DashSet</a>&lt;K, S&gt;"],["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>, V, S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"dashmap/struct.DashMap.html\" title=\"struct dashmap::DashMap\">DashMap</a>&lt;K, V, S&gt;"]],
"hashbrown":[["impl&lt;K, V, S, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/core/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"hashbrown/hash_map/struct.HashMap.html\" title=\"struct hashbrown::hash_map::HashMap\">HashMap</a>&lt;K, V, S, A&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    A: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> + Allocator + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</span>"],["impl&lt;T, S, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"hashbrown/hash_set/struct.HashSet.html\" title=\"struct hashbrown::hash_set::HashSet\">HashSet</a>&lt;T, S, A&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    A: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> + Allocator + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</span>"]],
"indexmap":[["impl&lt;T, S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"indexmap/set/struct.IndexSet.html\" title=\"struct indexmap::set::IndexSet\">IndexSet</a>&lt;T, S&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"],["impl&lt;K, V, S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"indexmap/map/struct.IndexMap.html\" title=\"struct indexmap::map::IndexMap\">IndexMap</a>&lt;K, V, S&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"]],
"linked_hash_map":[["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>, V, S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"linked_hash_map/struct.LinkedHashMap.html\" title=\"struct linked_hash_map::LinkedHashMap\">LinkedHashMap</a>&lt;K, V, S&gt;"]],
"onig":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.SyntaxOperator.html\" title=\"struct onig::SyntaxOperator\">SyntaxOperator</a>&gt; for <a class=\"struct\" href=\"onig/struct.SyntaxOperator.html\" title=\"struct onig::SyntaxOperator\">SyntaxOperator</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.RegexOptions.html\" title=\"struct onig::RegexOptions\">RegexOptions</a>&gt; for <a class=\"struct\" href=\"onig/struct.RegexOptions.html\" title=\"struct onig::RegexOptions\">RegexOptions</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.SyntaxBehavior.html\" title=\"struct onig::SyntaxBehavior\">SyntaxBehavior</a>&gt; for <a class=\"struct\" href=\"onig/struct.SyntaxBehavior.html\" title=\"struct onig::SyntaxBehavior\">SyntaxBehavior</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.SearchOptions.html\" title=\"struct onig::SearchOptions\">SearchOptions</a>&gt; for <a class=\"struct\" href=\"onig/struct.SearchOptions.html\" title=\"struct onig::SearchOptions\">SearchOptions</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.MetaCharType.html\" title=\"struct onig::MetaCharType\">MetaCharType</a>&gt; for <a class=\"struct\" href=\"onig/struct.MetaCharType.html\" title=\"struct onig::MetaCharType\">MetaCharType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"onig/struct.TraverseCallbackAt.html\" title=\"struct onig::TraverseCallbackAt\">TraverseCallbackAt</a>&gt; for <a class=\"struct\" href=\"onig/struct.TraverseCallbackAt.html\" title=\"struct onig::TraverseCallbackAt\">TraverseCallbackAt</a>"]],
"plist":[["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.71.1/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt;, V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"plist/enum.Value.html\" title=\"enum plist::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"plist/dictionary/struct.Dictionary.html\" title=\"struct plist::dictionary::Dictionary\">Dictionary</a>"]],
"proc_macro2":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"enum\" href=\"proc_macro2/enum.TokenTree.html\" title=\"enum proc_macro2::TokenTree\">TokenTree</a>&gt; for <a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>&gt; for <a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>"]],
"regex_syntax":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"regex_syntax/hir/literal/struct.Literal.html\" title=\"struct regex_syntax::hir::literal::Literal\">Literal</a>&gt; for <a class=\"struct\" href=\"regex_syntax/hir/literal/struct.Seq.html\" title=\"struct regex_syntax::hir::literal::Seq\">Seq</a>"]],
"rustix":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>"]],
"serde_json":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"struct\" href=\"https://doc.rust-lang.org/1.71.1/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>, <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>)&gt; for <a class=\"struct\" href=\"serde_json/struct.Map.html\" title=\"struct serde_json::Map\">Map</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.71.1/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>, <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;"],["impl&lt;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>"],["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.71.1/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt;, V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>"]],
"smallvec":[["impl&lt;A: <a class=\"trait\" href=\"smallvec/trait.Array.html\" title=\"trait smallvec::Array\">Array</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;&lt;A as <a class=\"trait\" href=\"smallvec/trait.Array.html\" title=\"trait smallvec::Array\">Array</a>&gt;::<a class=\"associatedtype\" href=\"smallvec/trait.Array.html#associatedtype.Item\" title=\"type smallvec::Array::Item\">Item</a>&gt; for <a class=\"struct\" href=\"smallvec/struct.SmallVec.html\" title=\"struct smallvec::SmallVec\">SmallVec</a>&lt;A&gt;"]],
"syn":[["impl&lt;T, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"syn/punctuated/struct.Punctuated.html\" title=\"struct syn::punctuated::Punctuated\">Punctuated</a>&lt;T, P&gt;<span class=\"where fmt-newline\">where\n    P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"],["impl&lt;T, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"enum\" href=\"syn/punctuated/enum.Pair.html\" title=\"enum syn::punctuated::Pair\">Pair</a>&lt;T, P&gt;&gt; for <a class=\"struct\" href=\"syn/punctuated/struct.Punctuated.html\" title=\"struct syn::punctuated::Punctuated\">Punctuated</a>&lt;T, P&gt;"]],
"syntect":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"syntect/highlighting/struct.FontStyle.html\" title=\"struct syntect::highlighting::FontStyle\">FontStyle</a>&gt; for <a class=\"struct\" href=\"syntect/highlighting/struct.FontStyle.html\" title=\"struct syntect::highlighting::FontStyle\">FontStyle</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()