(function() {var implementors = {
"dashmap":[["impl&lt;'a, K, V: 'a, S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>, Q&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.reference.html\">&amp;Q</a>&gt; for &amp;'a <a class=\"struct\" href=\"dashmap/struct.DashMap.html\" title=\"struct dashmap::DashMap\">DashMap</a>&lt;K, V, S&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/borrow/trait.Borrow.html\" title=\"trait core::borrow::Borrow\">Borrow</a>&lt;Q&gt; + 'a + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    Q: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</span>"]],
"hashbrown":[["impl&lt;T, S, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;&amp;<a class=\"struct\" href=\"hashbrown/hash_set/struct.HashSet.html\" title=\"struct hashbrown::hash_set::HashSet\">HashSet</a>&lt;T, S, A&gt;&gt; for &amp;<a class=\"struct\" href=\"hashbrown/hash_set/struct.HashSet.html\" title=\"struct hashbrown::hash_set::HashSet\">HashSet</a>&lt;T, S, A&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    A: Allocator + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</span>"]],
"indexmap":[["impl&lt;T, S1, S2&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;&amp;<a class=\"struct\" href=\"indexmap/set/struct.IndexSet.html\" title=\"struct indexmap::set::IndexSet\">IndexSet</a>&lt;T, S2&gt;&gt; for &amp;<a class=\"struct\" href=\"indexmap/set/struct.IndexSet.html\" title=\"struct indexmap::set::IndexSet\">IndexSet</a>&lt;T, S1&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,\n    S1: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    S2: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a>,</span>"]],
"onig":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.TraverseCallbackAt.html\" title=\"struct onig::TraverseCallbackAt\">TraverseCallbackAt</a>&gt; for <a class=\"struct\" href=\"onig/struct.TraverseCallbackAt.html\" title=\"struct onig::TraverseCallbackAt\">TraverseCallbackAt</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.SearchOptions.html\" title=\"struct onig::SearchOptions\">SearchOptions</a>&gt; for <a class=\"struct\" href=\"onig/struct.SearchOptions.html\" title=\"struct onig::SearchOptions\">SearchOptions</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.MetaCharType.html\" title=\"struct onig::MetaCharType\">MetaCharType</a>&gt; for <a class=\"struct\" href=\"onig/struct.MetaCharType.html\" title=\"struct onig::MetaCharType\">MetaCharType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.RegexOptions.html\" title=\"struct onig::RegexOptions\">RegexOptions</a>&gt; for <a class=\"struct\" href=\"onig/struct.RegexOptions.html\" title=\"struct onig::RegexOptions\">RegexOptions</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.SyntaxBehavior.html\" title=\"struct onig::SyntaxBehavior\">SyntaxBehavior</a>&gt; for <a class=\"struct\" href=\"onig/struct.SyntaxBehavior.html\" title=\"struct onig::SyntaxBehavior\">SyntaxBehavior</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"onig/struct.SyntaxOperator.html\" title=\"struct onig::SyntaxOperator\">SyntaxOperator</a>&gt; for <a class=\"struct\" href=\"onig/struct.SyntaxOperator.html\" title=\"struct onig::SyntaxOperator\">SyntaxOperator</a>"]],
"rustix":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>"]],
"syntect":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/bit/trait.BitAnd.html\" title=\"trait core::ops::bit::BitAnd\">BitAnd</a>&lt;<a class=\"struct\" href=\"syntect/highlighting/struct.FontStyle.html\" title=\"struct syntect::highlighting::FontStyle\">FontStyle</a>&gt; for <a class=\"struct\" href=\"syntect/highlighting/struct.FontStyle.html\" title=\"struct syntect::highlighting::FontStyle\">FontStyle</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()