"""DSL lexer + recursive-descent parser.

Implements the EBNF in `docs/user/query_language.md` §2. Hand-rolled, no
external parser dependency. Produces an AST of frozen dataclasses
exported below. Errors are surfaced as `QueryParseError(line, col,
message)`.

Grammar precedence (low → high):

    within > or > and > {while, then, because_of} > not > primary

Temporal operators are left-associative.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Union

from causal_ai_av.query.entities import (
    ENTITIES,
    TOP_LEVEL_ENTITIES,
    candidate_attribute_names,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QueryParseError(Exception):
    """Structured error with `(line, column, message)`."""

    def __init__(self, line: int, col: int, message: str) -> None:
        self.line = line
        self.col = col
        self.message = message
        super().__init__(f"line {line}, col {col}: {message}")


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrPredicate:
    entity: str
    path: tuple[str, ...]
    op: str
    value: object  # str | int | float | bool | frozenset


@dataclass(frozen=True)
class EntityClause:
    entity: str
    inner: "Expr"


@dataclass(frozen=True)
class And:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Or:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Not:
    expr: "Expr"


@dataclass(frozen=True)
class While:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Then:
    left: "Expr"
    right: "Expr"
    k: float


@dataclass(frozen=True)
class BecauseOf:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Within:
    window: "Expr"
    body: "Expr"


Expr = Union[
    AttrPredicate, EntityClause, And, Or, Not, While, Then, BecauseOf, Within
]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


_KEYWORDS = frozenset({
    "and", "or", "not", "in", "while", "then", "because_of", "within",
    "true", "false",
})

# Single-character punctuation tokens.
_PUNCT_SINGLE = {
    "(": "LPAREN",
    ")": "RPAREN",
    ":": "COLON",
    ".": "DOT",
    ",": "COMMA",
}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int


def _tokenize(src: str) -> list[Token]:
    out: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def _err(msg: str) -> QueryParseError:
        return QueryParseError(line, col, msg)

    while i < n:
        c = src[i]

        # Whitespace
        if c in " \t\r":
            i += 1
            col += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue

        # Comments
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue

        # Multi-char operators (order matters: longer first)
        if src.startswith("!=", i):
            out.append(Token("NE", "!=", line, col))
            i += 2
            col += 2
            continue
        if src.startswith(">=", i):
            out.append(Token("GE", ">=", line, col))
            i += 2
            col += 2
            continue
        if src.startswith("<=", i):
            out.append(Token("LE", "<=", line, col))
            i += 2
            col += 2
            continue

        if c == "=":
            out.append(Token("EQ", "=", line, col))
            i += 1
            col += 1
            continue
        if c == ">":
            out.append(Token("GT", ">", line, col))
            i += 1
            col += 1
            continue
        if c == "<":
            out.append(Token("LT", "<", line, col))
            i += 1
            col += 1
            continue

        # Punctuation
        if c in _PUNCT_SINGLE:
            out.append(Token(_PUNCT_SINGLE[c], c, line, col))
            i += 1
            col += 1
            continue

        # Numbers
        if c.isdigit():
            start_col = col
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            num_str = src[i:j]
            # Optional trailing "s" (seconds suffix in then(5s)).
            # Don't consume "s" if it's followed by an identifier char —
            # that would be the start of a name.
            had_suffix = False
            if j < n and src[j] == "s":
                k = j + 1
                if k >= n or not (src[k].isalnum() or src[k] == "_" or src[k] == ":"):
                    had_suffix = True
                    j += 1
            out.append(Token("NUMBER", num_str, line, start_col))
            col += (j - i)
            if had_suffix:
                # Note: we silently discard the "s" — it's just a unit
                # hint, the number is always seconds.
                pass
            i = j
            continue

        # Identifiers / keywords / qualified literals (oxd:Pedestrian etc.)
        if c.isalpha() or c == "_":
            start_col = col
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            # Optional qualified suffix: ":" + ident chars (oxd:Walk).
            # We only consume the colon if the next char looks like an
            # ident-start. Otherwise leave the COLON as its own token (it
            # could be the `within W:` colon).
            if j < n and src[j] == ":":
                k = j + 1
                if k < n and (src[k].isalpha() or src[k] == "_"):
                    j += 1
                    while j < n and (src[j].isalnum() or src[j] == "_"):
                        j += 1
            ident = src[i:j]
            lower = ident.lower()
            if lower in _KEYWORDS:
                out.append(Token(lower.upper(), ident, line, start_col))
            elif lower in ("true", "false"):
                out.append(Token("BOOL", ident, line, start_col))
            else:
                out.append(Token("IDENT", ident, line, start_col))
            col += (j - i)
            i = j
            continue

        # String literals (double-quoted) — used rarely but the spec
        # shows `clip.id = "00f2c7d3..."` in §3.11.
        if c == '"':
            start_col = col
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\n":
                    raise QueryParseError(line, col, "unterminated string literal")
                j += 1
            if j >= n:
                raise QueryParseError(line, col, "unterminated string literal")
            out.append(Token("IDENT", src[i + 1 : j], line, start_col))
            col += (j - i + 1)
            i = j + 1
            continue

        raise _err(f"unexpected character {c!r}")

    out.append(Token("EOF", "", line, col))
    return out


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _did_you_mean(name: str, options: list[str]) -> str:
    matches = difflib.get_close_matches(name.lower(), [o.lower() for o in options], n=3)
    if matches:
        return f" — did you mean: {', '.join(matches)}?"
    if options:
        sample = ", ".join(sorted(options)[:8])
        return f" (known: {sample}{'…' if len(options) > 8 else ''})"
    return ""


class _Parser:
    def __init__(self, tokens: list[Token], src: str) -> None:
        self.toks = tokens
        self.src = src
        self.pos = 0
        # Stack of bound entity names: when parsing the inside of
        # `agent(...)`, the top of the stack is "agent" — so
        # unqualified `type = ped` is rewritten as `agent.type = ped`.
        self._bound: list[str] = []

    # -- token helpers -----------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        return self.toks[min(self.pos + offset, len(self.toks) - 1)]

    def _advance(self) -> Token:
        t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def _check(self, *kinds: str) -> bool:
        return self._peek().kind in kinds

    def _match(self, *kinds: str) -> Token | None:
        if self._check(*kinds):
            return self._advance()
        return None

    def _expect(self, kind: str, what: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise QueryParseError(
                tok.line, tok.col,
                f"expected {what}, found {tok.kind!r} ({tok.value!r})",
            )
        return self._advance()

    def _err_at(self, tok: Token, message: str) -> QueryParseError:
        return QueryParseError(tok.line, tok.col, message)

    # -- grammar -----------------------------------------------------------

    def parse_query(self) -> Expr:
        expr = self.parse_expression()
        end = self._peek()
        if end.kind != "EOF":
            raise self._err_at(
                end, f"trailing tokens after expression: {end.value!r}"
            )
        return expr

    def parse_expression(self) -> Expr:
        if self._check("WITHIN"):
            return self.parse_within()
        return self.parse_or()

    def parse_within(self) -> Expr:
        self._expect("WITHIN", "'within'")
        window = self.parse_primary()
        self._expect("COLON", "':' after within-window")
        body = self.parse_expression()
        return Within(window=window, body=body)

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self._match("OR"):
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_temporal()
        while self._match("AND"):
            right = self.parse_temporal()
            left = And(left, right)
        return left

    def parse_temporal(self) -> Expr:
        left = self.parse_unary()
        while True:
            if self._match("WHILE"):
                right = self.parse_unary()
                left = While(left, right)
            elif self._check("THEN"):
                self._advance()
                k = 0.0
                if self._match("LPAREN"):
                    num = self._expect("NUMBER", "number")
                    k = float(num.value)
                    self._expect("RPAREN", "')' after then(K)")
                right = self.parse_unary()
                left = Then(left, right, k)
            elif self._match("BECAUSE_OF"):
                right = self.parse_unary()
                left = BecauseOf(left, right)
            else:
                break
        return left

    def parse_unary(self) -> Expr:
        if self._match("NOT"):
            inner = self.parse_unary()
            return Not(inner)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self._peek()
        if tok.kind == "LPAREN":
            self._advance()
            expr = self.parse_expression()
            self._expect("RPAREN", "')'")
            return expr
        if tok.kind != "IDENT":
            raise self._err_at(
                tok,
                f"expected entity name or '(' here, found {tok.kind} ({tok.value!r})",
            )

        # Capture the identifier as the head of a path. Then:
        #   - if followed by '(' → entity clause
        #   - else read dotted path → attribute predicate
        first = self._advance()
        head = first.value.lower()

        # If we're inside an entity clause and the head matches a
        # sub-entity attribute of the bound entity (e.g. `action` inside
        # `agent(...)`), and it's followed by '(', treat as a
        # sub-entity clause `agent.action(...)`.
        bound = self._bound[-1] if self._bound else None
        if bound is not None and self._check("LPAREN"):
            parent_desc = ENTITIES.get(bound)
            if parent_desc is not None:
                attr = parent_desc.attributes.get(head)
                if attr is not None and attr.kind == "sub_entity":
                    return self._parse_sub_entity_clause(first, bound, head)

        if self._check("LPAREN"):
            return self._parse_entity_clause(first, head)

        # Build dotted attribute path
        path: list[str] = [head]
        while self._match("DOT"):
            nxt = self._expect("IDENT", "attribute name after '.'")
            path.append(nxt.value.lower())

        # If the head isn't a known top-level entity but we have a bound
        # entity context whose attribute table contains it, prepend
        # the bound entity name. This implements the "no <entity>.
        # prefix needed inside an entity clause" rule (spec §3.3).
        if bound is not None and head not in TOP_LEVEL_ENTITIES:
            parent_desc = ENTITIES.get(bound)
            if parent_desc is not None and head in parent_desc.attributes:
                path = [bound] + path

        # If this looks like just one ident with no path and we hit a
        # paren without a dot, it would be the entity-clause branch
        # above. So now we must see a comparison op.
        if not self._check("EQ", "NE", "GT", "GE", "LT", "LE", "IN"):
            tok = self._peek()
            raise self._err_at(
                tok,
                f"expected comparison op after attribute path "
                f"{'.'.join(path)!r}, found {tok.value!r}",
            )

        return self._build_attr_predicate(first, path)

    def _parse_entity_clause(self, head_tok: Token, head: str) -> Expr:
        if head not in TOP_LEVEL_ENTITIES:
            hint = _did_you_mean(head, list(TOP_LEVEL_ENTITIES))
            raise self._err_at(
                head_tok, f"unknown entity {head!r}{hint}"
            )
        self._expect("LPAREN", "'('")
        self._bound.append(head)
        try:
            inner = self._parse_inner_constraints()
        finally:
            self._bound.pop()
        self._expect("RPAREN", "')'")
        return EntityClause(entity=head, inner=inner)

    def _parse_sub_entity_clause(
        self, head_tok: Token, parent: str, sub: str
    ) -> Expr:
        """Inside `agent(...)`: parse `action(...)` as a sub-entity clause.

        The qualified name is `agent.action`; we push it onto the bound
        stack so deeper unqualified attributes resolve correctly.
        """
        qualified = f"{parent}.{sub}"
        if qualified not in ENTITIES:
            raise self._err_at(
                head_tok, f"unknown sub-entity {sub!r} on {parent!r}"
            )
        self._expect("LPAREN", "'('")
        self._bound.append(qualified)
        try:
            inner = self._parse_inner_constraints()
        finally:
            self._bound.pop()
        self._expect("RPAREN", "')'")
        return EntityClause(entity=qualified, inner=inner)

    def _parse_inner_constraints(self) -> Expr:
        """Body of an entity clause: one or more expressions separated by
        ',' (sugar for AND), or full expression syntax. The two styles
        from the spec (§3.3 examples) are interchangeable.
        """
        expr = self.parse_expression()
        while self._match("COMMA"):
            right = self.parse_expression()
            expr = And(expr, right)
        return expr

    def _build_attr_predicate(self, head_tok: Token, path: list[str]) -> Expr:
        entity = path[0]
        rest = tuple(path[1:])

        # If `entity` is a qualified sub-entity name like `agent.action`,
        # split it so `entity` stays the top-level kind and the rest of
        # the path absorbs the sub-segment.
        if "." in entity and entity in ENTITIES and entity not in TOP_LEVEL_ENTITIES:
            top, sub = entity.split(".", 1)
            entity = top
            rest = (sub,) + rest

        if entity not in TOP_LEVEL_ENTITIES:
            hint = _did_you_mean(entity, list(TOP_LEVEL_ENTITIES))
            raise self._err_at(head_tok, f"unknown entity {entity!r}{hint}")

        # Walk the path. We allow `agent.action.type` and `agent.action`
        # (latter only with a value-set comparison meaning "type matches").
        if rest:
            self._validate_path(head_tok, entity, rest)

        op_tok = self._advance()
        op = {
            "EQ": "=", "NE": "!=", "GT": ">", "GE": ">=",
            "LT": "<", "LE": "<=", "IN": "in",
        }[op_tok.kind]

        if op == "in":
            self._expect("LPAREN", "'(' after 'in'")
            atoms: list[object] = [self._parse_atom()]
            while self._match("COMMA"):
                atoms.append(self._parse_atom())
            self._expect("RPAREN", "')' to close value set")
            value: object = frozenset(atoms)
        else:
            value = self._parse_atom()

        # Sugar: `<entity>.<sub> = <value>` (rest = ('action',) or
        # ('prop',)) — rewrite to `<entity>.<sub>.type <op> <value>`.
        # This covers `ego.action = decel`, `agent.action in (stop, ...)`.
        if rest and len(rest) == 1:
            ent_desc = ENTITIES[entity]
            attr = ent_desc.attributes.get(rest[0])
            if attr is not None and attr.kind == "sub_entity":
                rest = rest + ("type",)

        return AttrPredicate(entity=entity, path=rest, op=op, value=value)

    def _validate_path(
        self, head_tok: Token, entity: str, rest: tuple[str, ...]
    ) -> None:
        """Walk the dotted path to verify each segment is a known
        attribute (or a sub-entity, in which case the next segment must
        be an attribute on that sub-entity).
        """
        current = entity
        for i, seg in enumerate(rest):
            desc = ENTITIES.get(current)
            if desc is None:
                raise self._err_at(
                    head_tok, f"no attribute path possible from {current!r}"
                )
            attr = desc.attributes.get(seg)
            if attr is None:
                cands = candidate_attribute_names(current)
                hint = _did_you_mean(seg, cands)
                raise self._err_at(
                    head_tok,
                    f"unknown attribute {seg!r} on {current!r}{hint}",
                )
            if attr.kind == "sub_entity" and i < len(rest) - 1:
                current = attr.sub_entity or current

    def _parse_atom(self) -> object:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._advance()
            txt = tok.value
            if "." in txt:
                return float(txt)
            return int(txt)
        if tok.kind == "BOOL":
            self._advance()
            return tok.value.lower() == "true"
        if tok.kind in ("IDENT", "TRUE", "FALSE"):
            self._advance()
            v = tok.value
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            return v  # raw — the engine resolves aliases per family
        raise self._err_at(tok, f"expected value atom, found {tok.kind} ({tok.value!r})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(text: str) -> Expr:
    """Parse a DSL query and return the AST root.

    Raises `QueryParseError` on lex or parse failure.
    """
    tokens = _tokenize(text)
    p = _Parser(tokens, text)
    return p.parse_query()


__all__ = [
    "parse",
    "QueryParseError",
    "AttrPredicate",
    "EntityClause",
    "And",
    "Or",
    "Not",
    "While",
    "Then",
    "BecauseOf",
    "Within",
    "Expr",
]
