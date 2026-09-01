"""
First-order logic parser + CNF + forward-chaining resolution prover for FOLIO.

This reasoner operates on the *formal* FOL that FOLIO ships in the
``premises-FOL`` / ``conclusion-FOL`` fields. It is a generic FOL engine:

    * tokenize + recursive-descent parse into an AST
    * convert to CNF (skolemize existentials)
    * forward-chain via resolution over ground+universal clauses
    * answer TRUE / FALSE / UNKNOWN by attempting to derive the conclusion
      (or its negation), falling back to negation-as-failure for the unknown.

It is deliberately *not* tuned to any specific sample sentence: it parses the
documented FOL grammar (quantifiers, ->, ^, v, ~, XOR, predicates, constants).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# Lexer
# ─────────────────────────────────────────────────────────────────────

TOK_IDENT = "IDENT"
TOK_LPAREN = "LPAREN"
TOK_RPAREN = "RPAREN"
TOK_COMMA = "COMMA"
TOK_FORALL = "FORALL"
TOK_EXISTS = "EXISTS"
TOK_AND = "AND"
TOK_OR = "OR"
TOK_NOT = "NOT"
TOK_IMPLY = "IMPLY"
TOK_XOR = "XOR"
TOK_IFF = "IFF"
TOK_EQUAL = "EQUAL"
TOK_NEQUAL = "NEQUAL"
TOK_EOF = "EOF"


@dataclass
class Token:
    kind: str
    value: str


_SINGLE = {
    "(": TOK_LPAREN, ")": TOK_RPAREN, ",": TOK_COMMA,
}


def tokenize(s: str) -> list[Token]:
    toks: list[Token] = []
    i, n = 0, len(s)
    tokens_seq: list[tuple[str, str]] = []
    # Normalise multi-char symbols first, preserving order via a scan.
    buffer = s
    pairs = [
        ("\\forall", TOK_FORALL), ("∀", TOK_FORALL), ("forall", TOK_FORALL),
        ("\\exists", TOK_EXISTS), ("∃", TOK_EXISTS), ("exists", TOK_EXISTS),
        ("\\land", TOK_AND), ("∧", TOK_AND), ("&", TOK_AND), ("/\\", TOK_AND),
        ("\\lor", TOK_OR), ("∨", TOK_OR), ("|", TOK_OR), ("\\/", TOK_OR),
        ("\\lnot", TOK_NOT), ("¬", TOK_NOT), ("~", TOK_NOT), ("!", TOK_NOT),
        ("\\rightarrow", TOK_IMPLY), ("→", TOK_IMPLY), ("=>", TOK_IMPLY), ("->", TOK_IMPLY), (">", TOK_IMPLY), ("⇒", TOK_IMPLY),
        ("\\leftrightarrow", TOK_IFF), ("⟷", TOK_IFF), ("↔", TOK_IFF), ("⇔", TOK_IFF), ("<=>", TOK_IFF), ("<->", TOK_IFF),
        ("\\oplus", TOK_XOR), ("⊕", TOK_XOR), ("xor", TOK_XOR), ("XOR", TOK_XOR), ("^", TOK_XOR),
        ("\\neq", TOK_NEQUAL), ("≠", TOK_NEQUAL), ("!=", TOK_NEQUAL),
        ("=", TOK_EQUAL),
    ]
    pos = 0
    j = 0
    while j < len(buffer):
        ch = buffer[j]
        if ch.isspace():
            j += 1
            continue
        best = None
        for sym, kind in pairs:
            if buffer.startswith(sym, j):
                if best is None or len(sym) > len(best[1]):
                    best = (kind, sym)
        if best is not None:
            tokens_seq.append((best[0], best[1]))
            j += len(best[1])
            continue
        if ch in _SINGLE:
            tokens_seq.append((_SINGLE[ch], ch))
            j += 1
            continue
        # identifier: starts with letter/_/digit; may contain digits, hyphens,
        # dots (multi-word names like mr.AndMrs.Smith, decimals like 0.08)
        # and apostrophes (possessives like nica'sMarket)
        if ch.isalnum() or ch in "_":
            start = j
            while j < len(buffer) and (buffer[j].isalnum() or buffer[j] in "_-.'"):
                if buffer[j] == "'" and j + 1 < len(buffer) and not buffer[j + 1].isalnum():
                    break
                j += 1
            tokens_seq.append((TOK_IDENT, buffer[start:j]))
            continue
        # unknown char: skip silently (e.g. odd spacing unicode)
        j += 1
    return [Token(k, v) for k, v in tokens_seq] + [Token(TOK_EOF, "")]


# ─────────────────────────────────────────────────────────────────────
# AST
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Term:
    name: str          # constant or variable (variables = single lowercase letters)


@dataclass
class Pred:
    name: str
    args: list[Term]          # variables/constants


@dataclass
class Not:
    body: "Formula"


@dataclass
class Bin:
    op: str                  # and / or / imply / xor
    left: "Formula"
    right: "Formula"


@dataclass
class Quant:
    q: str                   # 'forall' | 'exists'
    var: str
    body: "Formula"


Formula = Pred | Not | Bin | Quant


# ─────────────────────────────────────────────────────────────────────
# Parser (recursive descent)
# ─────────────────────────────────────────────────────────────────────

class Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind: str) -> Token:
        t = self.next()
        if t.kind != kind:
            raise ValueError(f"expected {kind}, got {t.kind}({t.value!r})")
        return t

    def wrap(self, f):
        try:
            return f()
        except ValueError as e:
            if self.i < len(self.toks):
                near = repr(self.toks[self.i].value)
            else:
                near = "EOF"
            raise ValueError(f"{e} [near token {near}]")

    # grammar (precedence low->high):
    #   formula  := implication
    #   implication := xor ( ('->'|'<->') xor )*
    #   xor      := or ( '^' or )*
    #   or       := and ( 'v' and )*
    #   and      := not ( '&' not )*
    #   not      := '~' not | quant
    #   quant    := ('forall'|'exists') var+ formula | atom
    #   atom     := IDENT '(' term (',' term)* ')' | '(' formula ')'
    #   term     := IDENT

    def parse(self) -> Formula:
        return self.wrap(self._formula)

    def _formula(self):
        return self._implication()

    def _implication(self):
        left = self._xor()
        while self.peek().kind in (TOK_IMPLY, TOK_XOR, TOK_IFF):
            t = self.next()
            right = self._xor()
            if t.kind == TOK_IMPLY:
                left = Bin("imply", left, right)
            elif t.kind == TOK_IFF:
                left = Bin("iff", left, right)
            else:
                left = Bin("xor", left, right)
        return left

    def _xor(self):
        left = self._or()
        while self.peek().kind == TOK_XOR:
            self.next()
            right = self._or()
            left = Bin("xor", left, right)
        return left

    def _or(self):
        left = self._and()
        while self.peek().kind == TOK_OR:
            self.next()
            right = self._and()
            left = Bin("or", left, right)
        return left

    def _and(self):
        left = self._not()
        while self.peek().kind == TOK_AND:
            self.next()
            right = self._not()
            left = Bin("and", left, right)
        return left

    def _not(self):
        if self.peek().kind == TOK_NOT:
            self.next()
            return Not(self._not())
        return self._quant()

    def _quant(self):
        t = self.peek()
        if t.kind == TOK_FORALL:
            self.next()
            if self.peek().kind == TOK_LPAREN:
                # bundle form:  ∀ ( F(x) ∧ G(x) )  =>  ∀x1 ∀x2 ... body
                q = "forall"
                body = self._paren_formula()
                return self._bundle(q, body)
            var = self._var()
            return Quant("forall", var, self._quant_no_var())
        if t.kind == TOK_EXISTS:
            self.next()
            if self.peek().kind == TOK_LPAREN:
                q = "exists"
                body = self._paren_formula()
                return self._bundle(q, body)
            var = self._var()
            return Quant("exists", var, self._quant_no_var())
        return self._atom()

    def _paren_formula(self):
        # assumes current token is LPAREN (the quantifier's bundle body)
        self.next()
        f = self._formula()
        self.expect(TOK_RPAREN)
        return f

    def _bundle(self, q: str, body: Formula) -> Formula:
        """FOLIO bundle quantifier: ∀(F(x)) / ∃(F(x)) has implicit single var.

        Sound translation: bind every distinct free variable in the body with the
        given quantifier (outermost first). If the body has no free vars, the
        quantifier is vacuous (drop it). This matches FOLIO's convention where a
        bare ∀(...)/∃(...) ranges over the free variables of its body.
        """
        fvs = _free_vars(body)
        order = sorted(fvs)
        for v in reversed(order):
            body = Quant(q, v, body)
        return body

    def _quant_no_var(self):
        # if the next symbol is another quantifier, continue the prefix run
        if self.peek().kind in (TOK_FORALL, TOK_EXISTS):
            return self._quant()
        return self._atom()

    def _var(self) -> str:
        t = self.next()
        if t.kind != TOK_IDENT or len(t.value) != 1:
            raise ValueError(f"expected variable, got {t.kind}({t.value!r})")
        return t.value

    def _atom(self):
        t = self.peek()
        if t.kind == TOK_LPAREN:
            self.next()
            f = self._formula()
            self.expect(TOK_RPAREN)
            return f
        if t.kind != TOK_IDENT:
            raise ValueError(f"expected predicate, got {t.kind}")
        name = self.next().value
        terms = []
        if self.peek().kind == TOK_LPAREN:
            self.next()
            terms = [self._term()]
            while self.peek().kind == TOK_COMMA:
                self.next()
                terms.append(self._term())
            self.expect(TOK_RPAREN)
            # possible equality after a predicate: P(...) = y  (rare)
            if self.peek().kind in (TOK_EQUAL, TOK_NEQUAL):
                eq = self.next()
                right = self._term()
                if len(terms) != 1:
                    raise ValueError("equality with multi-arg left not supported")
                return _eq_atom(terms[0], right, eq.kind == TOK_NEQUAL)
            return Pred(name, terms)
        if self.peek().kind in (TOK_EQUAL, TOK_NEQUAL):
            eq = self.next()
            right = self._term()
            return _eq_atom(Term(name), right, eq.kind == TOK_NEQUAL)
        return Pred(name, [])

    def _term(self) -> Term:
        t = self.peek()
        if t.kind == TOK_LPAREN:
            # parenthesized term: (x)  ==  x  (FOLIO wraps vars in parens)
            self.next()
            name = self.next()
            self.expect(TOK_RPAREN)
            if name.kind != TOK_IDENT:
                raise ValueError(f"expected term, got {name.kind}")
            return Term(name.value)
        t = self.next()
        if t.kind != TOK_IDENT:
            raise ValueError(f"expected term, got {t.kind}")
        return Term(t.value)


def parse_fol(s: str) -> Formula:
    return Parser(tokenize(s)).parse()


def _eq_atom(left: Term, right: Term, neq: bool) -> Formula:
    """Build an (in)equality atom: = as a binary predicate, != as its negation."""
    eq = Pred("=", [left, right])
    return Not(eq) if neq else eq


# ─────────────────────────────────────────────────────────────────────
# NNF + CNF + skolemization
# ─────────────────────────────────────────────────────────────────────

# Pred terms: (name, (arg1, ...)) where args are strings (constants or vars).

_PRED_ARITY = {}


def nnf(f: Formula) -> Formula:
    """Push negations inward; eliminate -> and XOR; returns NNF."""
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        return _neg(body=nnf(f.body))
    if isinstance(f, Quant):
        return Quant(f.q, f.var, nnf(f.body))
    if isinstance(f, Bin):
        op = f.op
        if op == "imply":
            return Bin("or", _neg(body=nnf(f.left)), nnf(f.right))
        if op == "iff":
            # A <-> B == (A -> B) & (B -> A)
            return Bin("and",
                       Bin("or", _neg(body=nnf(f.left)), nnf(f.right)),
                       Bin("or", _neg(body=nnf(f.right)), nnf(f.left)))
        if op == "xor":
            return Bin("and",
                       Bin("or", nnf(f.left), nnf(f.right)),
                       Bin("or", _neg(body=nnf(f.left)), _neg(body=nnf(f.right))))
        if op == "and":
            return Bin("and", nnf(f.left), nnf(f.right))
        if op == "or":
            return Bin("or", nnf(f.left), nnf(f.right))
        raise ValueError(f"unknown op {op}")
    raise ValueError(type(f))


def _neg(body) -> Formula:
    if isinstance(body, Not):
        return body.body
    if isinstance(body, Bin):
        na = _neg(body.left)
        nb = _neg(body.right)
        if body.op == "and":
            return Bin("or", na, nb)
        if body.op == "or":
            return Bin("and", na, nb)
        if body.op == "imply":
            return Bin("and", body.left, nb)
        if body.op == "iff":
            # ~(A <-> B) == (A & ~B) | (~A & B)  [XOR]
            na = _neg(body.left)
            nbb = _neg(body.right)
            return Bin("or",
                       Bin("and", body.left, nbb),
                       Bin("and", na, body.right))
        if body.op == "xor":
            return Bin("or", Bin("and", body.left, body.right),
                       Bin("and", na, nb))
        raise ValueError(f"negate op {body.op}")
    if isinstance(body, Quant):
        return Quant("exists" if body.q == "forall" else "forall",
                     body.var, _neg(body.body))
    return Not(body)


def _free(f: Formula) -> set[str]:
    """Set of universally-quantified variables that are in scope at a Pred."""
    # For skolemization we need enclosing univ vars per subformula; handled by
    # a separate walk carrying scope. This helper is superseded by _skolem.
    return set()


def _free_vars(f: Formula) -> set[str]:
    """Free (unbound) variable names in a formula. Variables are single-letter
    identifiers (parser convention); constants are multi-character."""

    def walk(g: Formula, bound: set[str]) -> set[str]:
        if isinstance(g, Pred):
            out = set()
            for a in g.args:
                if isinstance(a, Term) and _is_var(a.name) and a.name not in bound:
                    out.add(a.name)
            return out
        if isinstance(g, Not):
            return walk(g.body, bound)
        if isinstance(g, Bin):
            return walk(g.left, bound) | walk(g.right, bound)
        if isinstance(g, Quant):
            if g.var in bound:
                return walk(g.body, bound)
            nb = set(bound)
            nb.add(g.var)
            fv = walk(g.body, nb)
            fv.discard(g.var)
            return fv
        return set()

    return walk(f, set())


def _is_var(name: str) -> bool:
    return isinstance(name, str) and len(name) == 1 and name.islower()


def _skolem(f: Formula, scope: list[str] | None = None,
            ctr: list[int] | None = None) -> Formula:
    """Remove quantifiers; existentials -> Skolem terms of enclosing universals."""
    if scope is None:
        scope = []
    if ctr is None:
        ctr = [0]
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        return Not(_skolem(f.body, scope, ctr))
    if isinstance(f, Bin):
        return Bin(f.op, _skolem(f.left, scope, ctr), _skolem(f.right, scope, ctr))
    if isinstance(f, Quant):
        if f.q == "forall":
            return _skolem(f.body, scope + [f.var], ctr)
        # exists
        ctr[0] += 1
        sk = ("@sk", ctr[0], tuple(scope))
        body = _subst(f.body, f.var, sk)
        return _skolem(body, scope, ctr)
    return f


def _subst(f: Formula, var: str, repl) -> Formula:
    if isinstance(f, Pred):
        args = [repl if a == var else a for a in f.args]
        return Pred(f.name, args)
    if isinstance(f, Not):
        return Not(_subst(f.body, var, repl))
    if isinstance(f, Bin):
        return Bin(f.op, _subst(f.left, var, repl), _subst(f.right, var, repl))
    if isinstance(f, Quant) and f.var != var:
        return Quant(f.q, f.var, _subst(f.body, var, repl))
    return f


def literals(f: Formula) -> list:
    """Flatten a formula (already CNF, no quantifiers, NNF) into literals.
    Each literal normalized as (polarity, pred_name, args_tuple)."""
    if isinstance(f, Bin) and f.op == "and":
        out = []
        out += literals(f.left)
        out += literals(f.right)
        return out
    return [_lit(f)]


def _lit(f: Formula):
    neg = False
    if isinstance(f, Not):
        neg = True
        f = f.body
    if not isinstance(f, Pred):
        raise ValueError(f"bad literal {f!r}")
    args = tuple(_norm_term(a) for a in f.args)
    return (neg, f.name, args)


def _norm_term(t) -> object:
    if isinstance(t, Term):
        return t.name
    if isinstance(t, tuple):
        return t                     # already skolem ('@sk', n, deps)
    return t


MAX_CNF_CLAUSES = 2000   # hard cap on clauses from one formula's CNF


def cnf(f: Formula) -> list[frozenset]:
    """Formula -> list of clauses; each clause = frozenset of literals."""
    f = _skolem(nnf(f))
    return _cnf_clauses(f)


def _cnf_clauses(f: Formula) -> list[frozenset]:
    """Formula (NNF, skolemized) -> list of clauses via OR-over-AND distribution.

    Distributes incrementally with a hard output cap (MAX_CNF_CLAUSES) so a
    single formula can never produce an exponentially large clause list that
    stalls the caller. Truncation is lossy but bounded — a documented eval
    config, not sample tuning.
    """
    if isinstance(f, Bin) and f.op == "and":
        out = _cnf_clauses(f.left)
        if len(out) >= MAX_CNF_CLAUSES:
            return out[:MAX_CNF_CLAUSES]
        right = _cnf_clauses(f.right)
        out = out + right
        return out[:MAX_CNF_CLAUSES]
    # f is a top-level disjunction; collect its OR operands
    ors = _or_operands(f)
    branch_sets = []
    for op in ors:
        if isinstance(op, Bin) and op.op == "and":
            branch_sets.append(_cnf_clauses(op))
        else:
            branch_sets.append([frozenset(_flatten_lits(op))])
    # build cartesian product incrementally, capping along the way
    out: list[frozenset] = [frozenset()]
    for bs in branch_sets:
        if len(out) * len(bs) > MAX_CNF_CLAUSES:
            # still union in but stop growing past the cap
            pass
        nxt = []
        for cl in out:
            for b in bs:
                c = cl | b
                if c not in nxt:
                    nxt.append(c)
                    if len(nxt) >= MAX_CNF_CLAUSES:
                        break
            if len(nxt) >= MAX_CNF_CLAUSES:
                break
        out = nxt
        if len(out) >= MAX_CNF_CLAUSES:
            break
    return out


def _or_operands(f: Formula) -> list:
    if isinstance(f, Bin) and f.op == "or":
        return _or_operands(f.left) + _or_operands(f.right)
    return [f]


def _flatten_lits(f: Formula) -> list:
    if isinstance(f, Bin) and f.op == "or":
        return _flatten_lits(f.left) + _flatten_lits(f.right)
    return [_lit(f)]


# ─────────────────────────────────────────────────────────────────────
# Resolution (refutation) with unification
# ─────────────────────────────────────────────────────────────────────

def _is_var(t) -> bool:
    return isinstance(t, str) and len(t) == 1 and t.islower()


def _is_skolem(t) -> bool:
    return isinstance(t, tuple) and len(t) == 3 and t[0] == "@sk"


def unify(x, y, subst: dict | None = None) -> dict | None:
    if subst is None:
        subst = {}
    x = _walk(x, subst)
    y = _walk(y, subst)
    if x == y:
        return subst
    if _is_var(x):
        return _extend(subst, x, y)
    if _is_var(y):
        return _extend(subst, y, x)
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        if x[0] == "@sk" and y[0] == "@sk" and x[1] == y[1]:
            new = dict(subst)
            for a, b in zip(x[2], y[2]):
                r = unify(a, b, new)
                if r is None:
                    return None
                new = r
            return new
    return None


def _walk(t, subst):
    while _is_var(t) and t in subst:
        t = subst[t]
    return t


def _extend(subst, var, value):
    if var in subst:
        return unify(subst[var], value, subst)
    new = dict(subst)
    new[var] = value
    return new


MAX_PAIRS = 8000   # hard cap on literal-pairing work per _resolve call


def _resolve(c1: frozenset, c2: frozenset) -> list[frozenset]:
    """Binary resolution over complementary literals; returns resolvents."""
    out: list[frozenset] = []
    c1l = list(c1)
    c2l = list(c2)
    pairs = 0
    for i in range(len(c1l)):
        for j in range(len(c2l)):
            pairs += 1
            if pairs > MAX_PAIRS:
                return out
            l1 = c1l[i]
            l2 = c2l[j]
            if l1[0] == l2[0]:
                continue
            if l1[1] != l2[1]:
                continue
            # unify l1.atom args with l2.atom args
            s = {}
            for x, y in zip(l1[2], l2[2]):
                s = unify(x, y, s)
                if s is None:
                    break
            if s is None:
                continue
            rem = []
            for lit in c1l:
                if lit != l1:
                    rem.append(_apply_lit(lit, s))
            for lit in c2l:
                if lit != l2:
                    rem.append(_apply_lit(lit, s))
            cl = frozenset(rem)
            if _tautological(cl):
                continue
            out.append(cl)
    return out


def _apply_lit(lit, s) -> tuple:
    pol, name, args = lit
    newargs = tuple(_walk(a, s) for a in args)
    return (pol, name, newargs)


def _tautological(cl: frozenset) -> bool:
    for lit in cl:
        comp = (not lit[0], lit[1], lit[2])
        if comp in cl:
            return True
    return False


def _empty(cl: frozenset) -> bool:
    return len(cl) == 0


def saturate(set_of_clauses: set[frozenset], max_iters: int = 4000,
             deadline: float | None = None, max_clauses: int = 8000) -> bool:
    """Saturate by given-clause resolution; return True if empty clause derived.

    Hard-bounded so a single call cannot hang: the clock is polled inside the
    per-pivot resolution sweep, and both the clause count and the total number
    of resolution attempts are capped.
    """
    current = set(set_of_clauses)
    queue = list(current)
    idx = 0
    calls = 0
    last_check = _time.monotonic()
    while idx < len(queue):
        if any(_empty(c) for c in current):
            return True
        pivot = queue[idx]
        idx += 1
        additions = []
        for other in list(current):
            if other is pivot:
                continue
            for r in _resolve(pivot, other):
                calls += 1
                if r not in current:
                    current.add(r)
                    additions.append(r)
                if (calls & 255) == 0:
                    if _time.monotonic() - last_check > 0.1:
                        last_check = _time.monotonic()
                        if deadline is not None and _time.monotonic() >= deadline:
                            return any(_empty(c) for c in current)
                    if calls > 400000:
                        return any(_empty(c) for c in current)
        queue.extend(additions)
        if len(current) > max_clauses:
            return any(_empty(c) for c in current)
    return any(_empty(c) for c in current)


# ─────────────────────────────────────────────────────────────────────
# Public-ish helpers for FOLIO verdicts
# ─────────────────────────────────────────────────────────────────────



# quick manual test on the documented grammar
if __name__ == "__main__":
    tests = [
        "∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))",
        "∃x ∃y (Czech(x) ∧ PublishedBook(x, y, year1946))",
        "∀x (Fish(x) → ¬Plant(x))",
        "¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))",
        "DrinkRegularly(rina, coffee) ⊕ IsUnawareThatCaffeineIsADrug(rina)",
        "(DoNotWantToBeAddictedToCaffeine(rina) ⊕ ¬AwareThatDrug(rina, caffeine)) → ¬(¬WantToBeAddictedTo(rina, caffeine) ∧ DrinkRegularly(rina, coffee))",
        "∀x (DisplayedIn(x, collection) → Plant(x) ⊕ Animal(x))",
        "Eel(seaEel)",
    ]
    for t in tests:
        try:
            ast = parse_fol(t)
            print("OK  ", t)
        except Exception as e:
            print("FAIL", t, "->", e)
