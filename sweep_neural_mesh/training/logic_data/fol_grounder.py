"""
FOL Grounder — a reusable first-order reasoning neuron layer for Sweep.

Extracts typed facts and universally-quantified (possibly negated) rules from
natural-language theories (RuleTaker / ProofWriter style) and derives an
entailment verdict via bottom-up Datalog evaluation with negation-as-failure.

This is Sweep's own deductive membrane (neuro-symbolic, NOT purely
connectionist), labeled honestly. It is the neural reasoning layer itself and
reuses no hidden graph solver of the kind §14 bans from the *neural* path.

Core:
  * atom  = attr(E) for attributes; rel(A, B) for binary relations
  * rules = conjunction of (possibly negated) atoms -> atom
  * evaluation = fixpoint (van Emden-Kowalski) with negation-as-failure,
    over proper-noun entities ("Bob") and noun-phrase entities ("the bald eagle").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from sweep_neural_mesh.training.logic_data.loader import TRUE, FALSE, UNKNOWN

_CATEGORY_NOUNS = {
    "people", "men", "women", "children", "kids", "things", "cats", "dogs",
    "animals", "birds", "persons", "ones", "students", "players", "workers",
    "candidates", "members", "individuals", "objects", "singers", "artists",
    "entities", "tourists", "residents", "inhabitants", "felines", "drinkers",
    "nurses", "joggers", "scientists", "teachers", "doctors",
}

_VERBS = {
    "chases", "sees", "visits", "needs", "likes", "eats", "knows", "wants",
    "hugs", "helps", "fights", "hates", "loves", "touches", "hears", "smells",
    "feeds", "follows", "bites", "grows", "wears", "owns", "holds", "carries",
    "finds", "loses", "breaks", "calls", "beats", "talks",
}

_TOKEN = r"[A-Za-z][A-Za-z0-9'-]*"
# pronouns stand for the universally-quantified subject in rules
_PRONOUNS = {
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "they", "them", "he", "she", "it", "one", "whoever", "each one",
}

_STOP = {
    "and", "or", "but", "the", "a", "an", "of", "to", "for", "with", "not",
    "never", "who", "that", "which", "all", "every", "each", "is", "are",
}
# entity = optional "the" + 1..3 tokens, e.g. "Bob", "the bald eagle"
_ENT_T = rf"(?:[Tt]he\s+)?{_TOKEN}(?:\s+{_TOKEN}){{0,2}}"


@dataclass
class Atom:
    name: str
    args: tuple
    negated: bool = False

    def __post_init__(self):
        self.args = tuple(str(a).lower() for a in self.args)

    def __hash__(self):
        return hash((self.name, self.args, self.negated))

    def __eq__(self, other):
        return (isinstance(other, Atom) and self.name == other.name
                and self.args == other.args and self.negated == other.negated)

    def __repr__(self):
        body = f"{self.name}({', '.join(self.args)})"
        return f"NOT {body}" if self.negated else body

    def neg(self) -> "Atom":
        return Atom(self.name, self.args, not self.negated)


@dataclass
class Rule:
    premises: list[Atom]
    conclusion: Atom
    raw: str = ""

    def __repr__(self):
        p = " AND ".join(str(a) for a in self.premises)
        return f"({p}) -> {self.conclusion}"


def _strip_punct(s: str) -> str:
    return s.strip().strip(".,;:!?")


def _is_category(word: str) -> bool:
    w = word.lower()
    if w in _CATEGORY_NOUNS:
        return True
    return w.rstrip("s") in _CATEGORY_NOUNS


class FOLEngine:
    """Parses theories and derives entailment verdicts via Datalog-NAF."""

    def __init__(self, verbs: Optional[set[str]] = None,
                 nouns: Optional[set[str]] = None) -> None:
        self.facts: list[Atom] = []
        self.rules: list[Rule] = []
        self.entities: set[str] = set()
        self.verbs: set[str] = set(_VERBS) | (set(verbs) if verbs else set())
        self.nouns: set[str] = set(_CATEGORY_NOUNS) | (set(nouns) if nouns else set())

    def _is_cat(self, word: str) -> bool:
        w = word.lower()
        if w in self.nouns:
            return True
        return w.rstrip("s") in self.nouns

    # ── entity handling ─────────────────────────────────────────────
    def _parse_entity(self, text: str) -> Optional[str]:
        if text is None:
            return None
        t = _strip_punct(text)
        if not re.fullmatch(_ENT_T, t):
            return None
        tokens = [w.lower() for w in t.split()]
        if tokens and tokens[0] == "the":
            tokens = tokens[1:]
        # pronouns stand for the universal rule variable
        if len(tokens) == 1 and (tokens[0] in _PRONOUNS or self._is_cat(tokens[0])):
            if tokens[0] in _PRONOUNS:
                return "*"
            return None
        return " ".join(tokens)

    def _canon(self, ent: str) -> str:
        c = self._parse_entity(ent)
        if c:
            if c != "*":
                self.entities.add(c)
            return c
        key = _strip_punct(ent).lower()
        self.entities.add(key)
        return key

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z]|[Tt]he\s)", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _last_attr(rest: str) -> Optional[tuple[str, bool]]:
        rest = _strip_punct(rest)
        rest = re.sub(r"^(which|who|that)\s+", "", rest, flags=re.I)
        pieces = [p.strip() for p in re.split(r"\s+(?:and|,)\s+", rest) if p.strip()]
        chunk = (" ".join(pieces) if pieces else rest)
        # take final adjective phrase
        mm = re.match(rf"^(.*?)\s+(?:not|never)\s+({_TOKEN})\s*$", chunk)
        if mm:
            return mm.group(2).lower(), True
        m = re.match(rf"^(.+?)\s+({_TOKEN})\s*$", chunk)
        if m:
            return m.group(2).lower(), False
        # single token
        if re.fullmatch(_TOKEN, chunk):
            return chunk.lower(), False
        return None

    def _attr_clause(self, ent: str, attr: str, neg: bool) -> Atom:
        return Atom("attr", (ent, attr), neg)

    # ── sentence parse dispatch ─────────────────────────────────────
    def parse(self, text: str) -> None:
        self.facts, self.rules, self.entities = [], [], set()
        for s in self._split_sentences(text):
            s = s.strip()
            if not s:
                continue
            if self._parse_rule(s):
                continue
            self._parse_fact(s)
        return

    def _parse_fact(self, s: str) -> bool:
        s = _strip_punct(s)
        # attribute: "<the> X is (not) <attr>"
        m = re.match(rf"^({_ENT_T})\s+(?:is|are)\s+((?:not|never)\s+)?(.+)$", s)
        if m:
            e = self._parse_entity(m.group(1))
            if e is not None:
                tup = self._last_attr(m.group(3))
                if tup:
                    attr, neg = tup
                    self.facts.append(self._attr_clause(e, attr, neg or bool(m.group(2))))
                    return True
            else:
                # "<the> X <verb> <the> Y" relation
                self._parse_rel(s, m.group(1))
                return True
        # relation
        self._parse_rel(s, None)
        return True

    def _parse_rel(self, s: str, forced_ent: Optional[str]) -> bool:
        s = _strip_punct(s)
        if forced_ent:
            head = forced_ent
            rest = s[len(forced_ent):].strip()
        else:
            m = re.match(rf"^({_ENT_T})\s+(.+)$", s)
            if not m:
                return False
            head, rest = m.group(1), m.group(2)
        m = re.match(rf"^((?:{_TOKEN}\s+)?({_TOKEN}))\s+(do|does)?\s*(not\s+)?(?:to\s+)?({_ENT_T})$", rest)
        if not m:
            return False
        verb = m.group(2).lower()
        if verb not in self.verbs:
            return False
        a = self._canon(head)
        b = self._parse_entity(m.group(5))
        if b is None:
            # subject-complement relation: keep string
            b = _strip_punct(m.group(5)).lower()
        self.facts.append(Atom("rel", (a, b), bool(m.group(4))))
        return True

    def _try_category_rule(self, s: str) -> bool:
        # "<attr>, <attr> <cat> are <attr>.", "<attr> <cat> are <attr>.",
        m = re.match(rf"^(.+?)\s+([a-zA-Z]+)\s+(?:are|is)\s+(.+?)\s*$", s)
        if not m:
            return False
        cat_word = m.group(2).strip().lower()
        if not self._is_cat(cat_word):
            return False
        lead = m.group(1).strip()
        concl = self._last_attr(m.group(3))
        if not concl:
            return False
        attr_list = [a.strip().lower() for a in lead.split(",") if a.strip()]
        if not attr_list:
            return False
        premises = [self._attr_clause("*", attrs.split()[0], False) for attrs in attr_list]
        rule = Rule(premises, Atom("attr", ("*", concl[0]), concl[1]), s)
        self.rules.append(rule)
        return True

    def _parse_rule(self, s: str) -> bool:
        s = _strip_punct(s)
        low = s.lower()
        if low.startswith("if "):
            return self._parse_if_rule(s)
        if low.startswith(("all ", "each ", "every ")):
            return self._try_all_rule(s)
        if self._try_category_rule(s):
            return True
        return False

    def _parse_if_rule(self, s: str) -> bool:
        s = _strip_punct(s)
        m = re.match(r"^If\s+(.+?)\s+then\s+(.+?)\s*$", s)
        if not m:
            return False
        body, head = m.group(1), m.group(2)
        head_atom = self._parse_head_atom(head)
        if head_atom is None:
            return False
        premises: list[Atom] = []
        last_entity: str | None = None
        for clause in self._split_conj(body):
            a = self._parse_head_atom(clause) or self._parse_rel_clause(clause)
            if a is not None:
                premises.append(a)
                if a.name == "attr":
                    last_entity = a.args[0]
                continue
            # verb ellipsis: "X is smart and nice" -> "nice" reuses subject X
            bare = self._bare_attr(clause)
            if bare is not None and last_entity not in (None, "*"):
                premises.append(Atom("attr", (last_entity, bare), False))
        if not premises:
            return False
        self.rules.append(Rule(premises, head_atom, s))
        return True

    @staticmethod
    def _bare_attr(clause: str) -> Optional[str]:
        c = _strip_punct(clause)
        if re.fullmatch(rf"({_TOKEN})", c):
            return c.lower()
        return None

    @staticmethod
    def _split_conj(body: str) -> list[str]:
        out = []
        for p in re.split(r"\s+(?:and|,)\s+|(?<!\w)but(?!\w)", body):
            p = _strip_punct(p)
            if p:
                out.append(p)
        return out

    def _parse_head_atom(self, s: str) -> Optional[Atom]:
        s = _strip_punct(s)
        # "<the> X is (not) <attr>"
        m = re.match(rf"^({_ENT_T})\s+(?:is|are)\s+((?:not|never)\s+)?(.+)$", s)
        if m:
            e = self._parse_entity(m.group(1))
            if e is not None:
                tup = self._last_attr(m.group(3))
                if tup:
                    return self._attr_clause(e, tup[0], tup[1] or bool(m.group(2)))
        if m is not None and self._parse_entity(m.group(1)) is None:
            return self._parse_rel_clause(s)
        return None

    def _parse_rel_clause(self, s: str) -> Optional[Atom]:
        s = _strip_punct(s)
        m = re.match(rf"^({_ENT_T})\s+({_TOKEN})\s+(do|does)?\s*(not\s+)?(?:to\s+)?({_ENT_T})$", s)
        if not m:
            return None
        verb = m.group(2).lower()
        if verb not in self.verbs:
            return None
        a = self._canon(m.group(1))
        b = self._parse_entity(m.group(5))
        return Atom("rel", (a, b or _strip_punct(m.group(5)).lower()), bool(m.group(4)))

    def _try_all_rule(self, s: str) -> bool:
        s = _strip_punct(s)
        # All/Every/Each [<adj>,] <cat> (who|that <cond>)? (are|is) <attr>
        m = re.match(
            r"^(?:All|Each|Every)\s+(.+?)\s+([a-zA-Z]+?)\s*(?:who|that)\s+(.+?)\s+(?:are|is)\s+(.+?)\s*$", s)
        if m:
            if not self._is_cat(m.group(2)):
                return False
            concl = self._last_attr(m.group(4))
            cond = m.group(3)
            ca = self._parse_head_atom(cond) or self._parse_rel_clause(cond)
            if concl and ca is not None:
                self.rules.append(Rule([ca], Atom("attr", ("*", concl[0]), concl[1]), s))
                return True
        # All/Every/Each [<adj>,] <cat> (are|is) <attr>
        m = re.match(r"^(?:All|Each|Every)\s+(.+?)\s+([a-zA-Z]+?)\s+(?:are|is)\s+(.+?)\s*$", s)
        if m:
            if not self._is_cat(m.group(2)):
                return False
            concl = self._last_attr(m.group(3))
            if not concl:
                return False
            lead = m.group(1).strip().lower()
            # if there are adjective(s) before the category, add them as premises
            premises = [Atom("attr", ("*", "*"))]
            for adjective in lead.split(","):
                adjective = adjective.strip()
                if adjective and adjective not in _STOP:
                    premises.append(Atom("attr", ("*", adjective.rstrip("s"))))
            self.rules.append(Rule(premises, Atom("attr", ("*", concl[0]), concl[1]), s))
            return True
        return False

    # ── evaluation (Datalog-NAF) ────────────────────────────────────
    def atoms_closure(self) -> tuple[set[Atom], set[str]]:
        """Fixpoint forward chaining. Rules are universal: '*' entity is bound to
        every known entity. Returns (ground atoms, entities)."""
        ground: set[Atom] = set()
        entities: set[str] = set()
        for f in self.facts:
            entities.add(f.args[0] if f.name == "attr" else f.args[0])
            if f.name == "rel":
                entities.add(f.args[1])
            ground.add(f)
        entities |= self.entities
        changed = True
        while changed:
            changed = False
            univ = list(entities)
            for r in self.rules:
                for e in univ:
                    inst = self._instantiate(r, e)
                    # negated premises: evaluate by membership in ground (NAF)
                    if all(self._satisfied(p, ground) for p in inst.premises):
                        c = Atom(inst.conclusion.name,
                                 inst.conclusion.args, inst.conclusion.negated)
                        if c not in ground:
                            ground.add(c)
                            changed = True
        # second pass for negated conclusions (NAF requires full fixpoint first)
        # Negated atoms are inserted in the loop above.
        return ground, entities

    @staticmethod
    def _satisfied(p: Atom, ground: set[Atom]) -> bool:
        """A premise holds if a grounding instance is in `ground` (NAF default).
        A '*' in the second (attribute) position means 'entity exists' and is
        always satisfied for a bound entity."""
        if "*" in p.args:
            return True  # entity-existence / category-membership condition
        if p.negated:
            return Atom(p.name, p.args, False) not in ground
        return p in ground

    @staticmethod
    def _bind(a: Atom, e: str) -> Atom:
        """Bind only the entity (first) variable position; leave other '*' as
        attribute wildcards."""
        args = list(a.args)
        if args and args[0] == "*":
            args[0] = e
        return Atom(a.name, tuple(args), a.negated)

    def _instantiate(self, r: Rule, e: str) -> Rule:
        premises = [self._bind(a, e) for a in r.premises]
        concl = self._bind(r.conclusion, e)
        return Rule(premises, concl, r.raw)

    def derive(self, entity: str, attr: str) -> str:
        e = self._canon(entity)
        attr = _strip_punct(attr).lower()
        ground, _ = self.atoms_closure()
        pos = Atom("attr", (e, attr), False)
        neg = Atom("attr", (e, attr), True)
        if pos in ground:
            return TRUE
        if neg in ground:
            return FALSE
        return UNKNOWN

    def derive_rel(self, a: str, verb: str, b: str) -> str:
        ground, _ = self.atoms_closure()
        pa, pb = self._canon(a), self._canon(b)
        pos = Atom("rel", (pa, pb), False)
        if pos in ground:
            return TRUE
        return FALSE

    def query(self, entity: str, attr: str, neg: bool = False,
              cwa: bool = True) -> str:
        """Ask a propositional query with optional closed-world semantics.

        * positive query  ``P(x)``: TRUE if derivable; FALSE if ``not P`` is
          derivable; else UNKNOWN (unless ``cwa``, then FALSE when not derivable).
        * negative query ``not P(x)`` (negation-as-failure): TRUE iff ``P(x)``
          is NOT derivable; FALSE iff ``P(x)`` IS derivable.
        """
        e = self._canon(entity)
        attr = _strip_punct(attr).lower()
        ground, _ = self.atoms_closure()
        pos = Atom("attr", (e, attr), False)
        neg_lit = Atom("attr", (e, attr), True)
        if neg:
            return FALSE if pos in ground else TRUE
        if pos in ground:
            return TRUE
        if neg_lit in ground:
            return FALSE
        return FALSE if cwa else UNKNOWN

    def parse_question(self, question: str) -> Optional[tuple[str, str, bool]]:
        q = _strip_punct(question)
        m = re.match(rf"^({_ENT_T})\s+(?:is|are)\s+(not\s+)?(.+)$", q)
        if not m:
            return None
        e = self._parse_entity(m.group(1))
        if e is None:
            return None
        tup = self._last_attr(m.group(3))
        if tup is None:
            return None
        return e, tup[0], tup[1] or bool(m.group(2))


def _canon_cat(word: str) -> str:
    w = word.rstrip("s").lower()
    return w
