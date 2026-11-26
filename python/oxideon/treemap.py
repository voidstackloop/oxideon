"""Hierarchical container traversal and mapping utilities (`oxideon.treemap` / `oxideon.utils.treemap`).

Provides structural flattening, reconstruction, multi-tree mapping, type-filtered mapping,
prefix broadcasting, structure assertions, and custom container node registration.
"""

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type


FlattenFn = Callable[[Any], Tuple[List[Any], Any]]
UnflattenFn = Callable[[Iterable[Any], Any], Any]


@dataclass(frozen=True)
class _NodeDef:
    typ: Type[Any]
    flatten_fn: FlattenFn
    unflatten_fn: UnflattenFn


REGISTERED_CONTAINERS: Dict[Type[Any], _NodeDef] = {}


def _is_namedtuple_instance(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(type(x), "_fields")


def _is_namedtuple_class(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields")


def register_tree_node(
    cls: Type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
) -> None:
    """Registers a custom container class `cls` as an internal tree node."""
    if not isinstance(cls, type):
        raise TypeError(f"register_tree_node expected a type, got {cls!r}")
    REGISTERED_CONTAINERS[cls] = _NodeDef(cls, flatten_fn, unflatten_fn)


def unregister_tree_node(cls: Type[Any]) -> None:
    """Removes a previously registered custom container class."""
    REGISTERED_CONTAINERS.pop(cls, None)


# Built-in container registrations
register_tree_node(list, lambda xs: (list(xs), None), lambda vals, _: list(vals))
register_tree_node(tuple, lambda xs: (list(xs), None), lambda vals, _: tuple(vals))
register_tree_node(dict, lambda d: (list(d.values()), list(d.keys())), lambda vals, keys: dict(zip(keys, vals)))
register_tree_node(OrderedDict, lambda d: (list(d.values()), list(d.keys())), lambda vals, keys: OrderedDict(zip(keys, vals)))
register_tree_node(
    defaultdict,
    lambda d: (list(d.values()), (d.default_factory, list(d.keys()))),
    lambda vals, ctx: defaultdict(ctx[0], zip(ctx[1], vals)),
)
register_tree_node(deque, lambda dq: (list(dq), dq.maxlen), lambda vals, maxlen: deque(vals, maxlen=maxlen))


class TreeSchema:
    """Describes the hierarchical container layout of a nested structure."""

    def __init__(
        self,
        typ: Optional[Type[Any]],
        context: Any,
        children_schemas: Sequence["TreeSchema"],
    ) -> None:
        self.type = typ
        self.context = context
        self.children_schemas: List[TreeSchema] = list(children_schemas)
        if typ is None:
            self.num_leaves = 1
            self.num_nodes = 1
            self.num_children = 0
        else:
            self.num_leaves = sum(c.num_leaves for c in self.children_schemas)
            self.num_nodes = 1 + sum(c.num_nodes for c in self.children_schemas)
            self.num_children = len(self.children_schemas)

    def is_leaf(self) -> bool:
        return self.type is None

    def unflatten(self, leaves: Iterable[Any]) -> Any:
        return tree_unflatten(leaves, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeSchema):
            return False
        return (
            self.type == other.type
            and self.context == other.context
            and self.children_schemas == other.children_schemas
        )

    def __repr__(self) -> str:
        if self.is_leaf():
            return "*"
        children_str = ", ".join(repr(c) for c in self.children_schemas)
        type_name = getattr(self.type, "__name__", str(self.type))
        return f"TreeSchema({type_name}, {self.context!r}, [{children_str}])"


class LeafSchema(TreeSchema):
    """Schema representing a single atomic leaf node."""

    def __init__(self) -> None:
        super().__init__(None, None, [])


def _get_node_type(x: Any) -> Optional[Type[Any]]:
    t = type(x)
    if t in REGISTERED_CONTAINERS:
        return t
    if _is_namedtuple_instance(x):
        return t
    return None


def _flatten_one(x: Any) -> Tuple[List[Any], Any, Type[Any]]:
    t = type(x)
    if t in REGISTERED_CONTAINERS:
        children, ctx = REGISTERED_CONTAINERS[t].flatten_fn(x)
        return list(children), ctx, t
    if _is_namedtuple_instance(x):
        return list(x), t, t
    raise TypeError(f"Not a registered tree container: {t}")


def _unflatten_one(typ: Type[Any], ctx: Any, children: List[Any]) -> Any:
    if typ in REGISTERED_CONTAINERS:
        return REGISTERED_CONTAINERS[typ].unflatten_fn(children, ctx)
    if _is_namedtuple_class(typ):
        return typ(*children)
    raise TypeError(f"Unknown tree container type: {typ}")


def is_nested(seq: Any) -> bool:
    """Returns True if `seq` is a registered nested container (strings and bytes are atomic leaves)."""
    return _get_node_type(seq) is not None


def tree_is_leaf(tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None) -> bool:
    """Returns True if `tree` is treated as a leaf node."""
    if is_leaf is not None and is_leaf(tree):
        return True
    return _get_node_type(tree) is None


def tree_flatten(
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Tuple[List[Any], TreeSchema]:
    """Flattens a nested structure into `(leaves, schema)`."""
    if tree_is_leaf(tree, is_leaf=is_leaf):
        return [tree], LeafSchema()

    children, ctx, typ = _flatten_one(tree)
    leaves: List[Any] = []
    child_schemas: List[TreeSchema] = []
    for ch in children:
        ch_leaves, ch_schema = tree_flatten(ch, is_leaf=is_leaf)
        leaves.extend(ch_leaves)
        child_schemas.append(ch_schema)
    return leaves, TreeSchema(typ, ctx, child_schemas)


def tree_unflatten(leaves: Iterable[Any], schema: TreeSchema) -> Any:
    """Reconstructs a nested structure from `leaves` and `schema`."""
    if not isinstance(schema, TreeSchema):
        raise TypeError(f"tree_unflatten expected a TreeSchema, got {type(schema).__name__}")
    flat = list(leaves)
    if len(flat) != schema.num_leaves:
        raise ValueError(f"tree_unflatten expected {schema.num_leaves} leaves, got {len(flat)}")
    it = iter(flat)

    def _build(s: TreeSchema) -> Any:
        if s.is_leaf():
            return next(it)
        assert s.type is not None
        built_children = [_build(cs) for cs in s.children_schemas]
        return _unflatten_one(s.type, s.context, built_children)

    return _build(schema)


def tree_leaves(
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> List[Any]:
    """Returns a flat list of leaves in `tree`."""
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf)
    return leaves


def tree_schema(
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> TreeSchema:
    """Returns the `TreeSchema` describing `tree`."""
    _, schema = tree_flatten(tree, is_leaf=is_leaf)
    return schema


def tree_map(
    fn: Callable[..., Any],
    tree: Any,
    *rests: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Applies `fn` across corresponding leaves of `tree` (and optional matching `*rests` trees)."""
    leaves, schema = tree_flatten(tree, is_leaf=is_leaf)
    if not rests:
        return tree_unflatten([fn(x) for x in leaves], schema)

    all_leaves = [leaves]
    for other in rests:
        other_leaves, other_schema = tree_flatten(other, is_leaf=is_leaf)
        if other_schema != schema:
            raise ValueError(f"tree_map requires identical tree schemas: expected {schema}, got {other_schema}")
        all_leaves.append(other_leaves)
    return tree_unflatten([fn(*args) for args in zip(*all_leaves)], schema)


def tree_map_(
    fn: Callable[[Any], Any],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Applies `fn` to every leaf of `tree` for in-place side effects and returns `tree`."""
    for leaf in tree_leaves(tree, is_leaf=is_leaf):
        fn(leaf)
    return tree


def _make_type_pred(type_or_types_or_pred: Any) -> Callable[[Any], bool]:
    if isinstance(type_or_types_or_pred, type) or (
        isinstance(type_or_types_or_pred, tuple)
        and all(isinstance(t, type) for t in type_or_types_or_pred)
    ):
        return lambda x: isinstance(x, type_or_types_or_pred)
    if callable(type_or_types_or_pred):
        return type_or_types_or_pred
    raise TypeError(f"Expected a type, tuple of types, or predicate, got {type_or_types_or_pred!r}")


def tree_map_only(
    type_or_types_or_pred: Any,
    fn: Callable[[Any], Any],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Applies `fn` only to leaves matching `type_or_types_or_pred`, leaving other leaves untouched."""
    pred = _make_type_pred(type_or_types_or_pred)
    return tree_map(lambda x: fn(x) if pred(x) else x, tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> bool:
    """Returns True if `pred(leaf)` holds for all leaves in `tree`."""
    return all(pred(x) for x in tree_leaves(tree, is_leaf=is_leaf))


def tree_any(
    pred: Callable[[Any], bool],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> bool:
    """Returns True if `pred(leaf)` holds for at least one leaf in `tree`."""
    return any(pred(x) for x in tree_leaves(tree, is_leaf=is_leaf))


def tree_all_only(
    type_or_types: Any,
    pred: Callable[[Any], bool],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> bool:
    """Returns True if `pred(leaf)` holds for all leaves matching `type_or_types`."""
    filt = _make_type_pred(type_or_types)
    return all(pred(x) for x in tree_leaves(tree, is_leaf=is_leaf) if filt(x))


def tree_any_only(
    type_or_types: Any,
    pred: Callable[[Any], bool],
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> bool:
    """Returns True if `pred(leaf)` holds for any leaf matching `type_or_types`."""
    filt = _make_type_pred(type_or_types)
    return any(pred(x) for x in tree_leaves(tree, is_leaf=is_leaf) if filt(x))


def broadcast_tree_prefix(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> List[Any]:
    """Broadcasts leaves of `prefix_tree` across matching subtrees of `full_tree`."""
    result: List[Any] = []

    def _walk(p: Any, f: Any) -> None:
        if tree_is_leaf(p, is_leaf=is_leaf):
            _, f_schema = tree_flatten(f, is_leaf=is_leaf)
            result.extend([p] * f_schema.num_leaves)
            return
        if tree_is_leaf(f, is_leaf=is_leaf):
            raise ValueError(f"prefix_tree has deeper nesting than full_tree at {p!r} vs {f!r}")
        p_children, p_ctx, p_typ = _flatten_one(p)
        f_children, f_ctx, f_typ = _flatten_one(f)
        if p_typ != f_typ or p_ctx != f_ctx or len(p_children) != len(f_children):
            raise ValueError(f"Mismatched container in broadcast_tree_prefix: {p_typ}({p_ctx}) vs {f_typ}({f_ctx})")
        for pc, fc in zip(p_children, f_children):
            _walk(pc, fc)

    _walk(prefix_tree, full_tree)
    return result


def flatten_nested(structure: Any) -> List[Any]:
    """Flattens a nested structure into a list of leaves (sorting standard `dict` keys deterministically)."""
    if tree_is_leaf(structure):
        return [structure]
    if isinstance(structure, dict) and not isinstance(structure, OrderedDict):
        out: List[Any] = []
        for k in sorted(structure.keys()):
            out.extend(flatten_nested(structure[k]))
        return out
    children, _, _ = _flatten_one(structure)
    out = []
    for ch in children:
        out.extend(flatten_nested(ch))
    return out


def assert_same_structure(
    nest1: Any,
    nest2: Any,
    check_types: bool = True,
) -> None:
    """Asserts two nested structures have identical shape.

    Raises `TypeError` when `check_types=True` and container types differ, and `ValueError`
    when arities, keys, or nesting depths differ.
    """
    n1_nested = is_nested(nest1)
    n2_nested = is_nested(nest2)
    if n1_nested != n2_nested:
        raise ValueError(f"Nested structures differ in depth: {nest1!r} vs {nest2!r}")
    if not n1_nested:
        return

    t1, t2 = type(nest1), type(nest2)
    if check_types:
        if _is_namedtuple_instance(nest1) or _is_namedtuple_instance(nest2):
            if t1 is not t2:
                raise TypeError(f"Namedtuple types differ: {t1} vs {t2}")
        elif isinstance(nest1, dict) or isinstance(nest2, dict):
            if t1 is not t2:
                raise TypeError(f"Mapping types differ: {t1} vs {t2}")
        elif t1 is not t2:
            raise TypeError(f"Sequence types differ: {t1} vs {t2}")
    else:
        if isinstance(nest1, dict) != isinstance(nest2, dict):
            raise ValueError(f"Cannot match mapping against sequence: {t1} vs {t2}")

    if isinstance(nest1, dict) and isinstance(nest2, dict):
        if set(nest1.keys()) != set(nest2.keys()):
            raise ValueError(f"Dictionary keys differ: {sorted(nest1.keys())} vs {sorted(nest2.keys())}")
        if isinstance(nest1, OrderedDict) and isinstance(nest2, OrderedDict) and check_types:
            if list(nest1.keys()) != list(nest2.keys()):
                raise ValueError(f"OrderedDict key order differs: {list(nest1.keys())} vs {list(nest2.keys())}")
        for k in nest1:
            assert_same_structure(nest1[k], nest2[k], check_types=check_types)
        return

    c1, _, _ = _flatten_one(nest1)
    c2, _, _ = _flatten_one(nest2)
    if len(c1) != len(c2):
        raise ValueError(f"Lengths differ: {len(c1)} vs {len(c2)}")
    for a, b in zip(c1, c2):
        assert_same_structure(a, b, check_types=check_types)


def pack_sequence_as(structure: Any, flat_sequence: Sequence[Any]) -> Any:
    """Packs `flat_sequence` into a nested structure matching `structure`."""
    flat = list(flat_sequence)
    expected = len(flatten_nested(structure))
    if len(flat) != expected:
        raise ValueError(f"Structure had {expected} leaves, but flat_sequence had {len(flat)} items.")
    it = iter(flat)

    def _pack(s: Any) -> Any:
        if not is_nested(s):
            return next(it)
        if isinstance(s, OrderedDict):
            return OrderedDict((k, _pack(s[k])) for k in s.keys())
        if isinstance(s, defaultdict):
            d = defaultdict(s.default_factory)
            for k in sorted(s.keys()):
                d[k] = _pack(s[k])
            return d
        if isinstance(s, dict):
            keys = sorted(s.keys())
            packed_pairs = {k: _pack(s[k]) for k in keys}
            return {k: packed_pairs[k] for k in s.keys()}
        children, ctx, typ = _flatten_one(s)
        return _unflatten_one(typ, ctx, [_pack(c) for c in children])

    return _pack(structure)


def map_structure(
    func: Callable[..., Any],
    *structures: Any,
    check_types: bool = True,
) -> Any:
    """Applies `func` to each leaf across one or more matching nested structures."""
    if not callable(func):
        raise TypeError(f"func must be callable, got {func!r}")
    if not structures:
        raise ValueError("Must provide at least one structure")
    first = structures[0]
    for other in structures[1:]:
        assert_same_structure(first, other, check_types=check_types)
    flat_lists = [flatten_nested(s) for s in structures]
    mapped = [func(*args) for args in zip(*flat_lists)]
    return pack_sequence_as(first, mapped)


__all__ = [
    "TreeSchema",
    "LeafSchema",
    "REGISTERED_CONTAINERS",
    "register_tree_node",
    "unregister_tree_node",
    "is_nested",
    "tree_is_leaf",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_schema",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "broadcast_tree_prefix",
    "flatten_nested",
    "assert_same_structure",
    "pack_sequence_as",
    "map_structure",
]
