---
title: "Style Guide for Python Coding"
date: 2020-01-02T14:58:43+01:00
categories: [Tech,Programming]
tags: [Python]
slug: "summary-style-python"
toc: true
displayCopyright: false
mermaid: false
---

This post contains summary of [Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/). And it helps you to write standard codes, name the variables, functions right, etc. ❤ <!--more-->

## Code Lay-out

Shortcut in VSCode: `Alt+Shift+F`

## String Quotes

[PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)

> A docstring is a string literal that occurs as the first statement in a module, function, class, or method definition. Such a docstring becomes the `__doc__` special attribute of that object.

- For consistency, always use `"""triple double quotes"""` around docstrings. 

- Use `r"""raw triple double quotes"""` if you use any backslashes in your docstrings. 

- For Unicode docstrings, use `u"""Unicode triple-quoted strings"""`.

### One-line Docstrings

```python
def kos_root():
    """Return the pathname of the KOS root directory."""
    global _kos_root
    if _kos_root: return _kos_root
    ...
```

- The one-line docstring should NOT be a "signature" reiterating the function/method parameters (which can be obtained by introspection). Don't do:

  ```python
  def function(a, b):
      """function(a, b) -> list"""
  ```

### Multi-line Docstrings

Multi-line docstrings consist of a summary line just like a one-line docstring, followed by a blank line, followed by a more elaborate description.

e.g. for a function (list each argument on a separate line):

```python
def complex(real=0.0, imag=0.0):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    if imag == 0.0 and real == 0.0:
        return complex_zero
    ...
```

## Comments

Comments that contradict the code are worse than no comments. Always make a priority of keeping the comments up-to-date when the code changes!

Comments should be complete sentences. The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).

Block comments generally consist of one or more paragraphs built out of complete sentences, with each sentence ending in a period.

You should use two spaces after a sentence-ending period in multi- sentence comments, except after the final sentence.

### Block Comments

Block comments generally apply to some (or all) code that follows them, and are indented to the same level as that code. Each line of a block comment starts with a `#` and a single space (unless it is indented text inside the comment).

Paragraphs inside a block comment are separated by a line containing a single `#`.

### Inline Comments

Use inline comments sparingly.

An inline comment is a comment on the same line as a statement. Inline comments should be separated by at least two spaces from the statement. They should start with a # and a single space.

Inline comments are unnecessary and in fact distracting if they state the obvious. Don't do this:

```python
x = x + 1                 # Increment x
```

But sometimes, this is useful:

```python
x = x + 1                 # Compensate for border
```

## Naming Conventions

### Overriding Principle

Names that are visible to the user as public parts of the API should **follow conventions that reflect usage** rather than implementation.

### Descriptive: Naming Styles

The following naming styles are commonly distinguished:

- `b` (single lowercase letter)

- `B` (single uppercase letter)

- `lowercase`

- `lower_case_with_underscores`

- `UPPERCASE`

- `UPPER_CASE_WITH_UNDERSCORES`

- `CapitalizedWords` (or CapWords, or CamelCase -- so named because of the bumpy look of its letters [[4\]](https://www.python.org/dev/peps/pep-0008/#id11)). This is also sometimes known as StudlyCaps.

  Note: When using acronyms in CapWords, capitalize all the letters of the acronym. Thus HTTPServerError is better than HttpServerError.

- `mixedCase` (differs from CapitalizedWords by initial lowercase character!)

### Prescriptive: Naming Conventions

#### Names to Avoid

Never use the characters 'l' (lowercase letter el), 'O' (uppercase letter oh), or 'I' (uppercase letter eye) as single character variable names.

#### Package and Module Names

`short`,`all-lowercase`,`1-word`

#### Class Names

`CapWords` (exception names and built-in constants)

#### Exception Names

`CapWords` with suffix `Error`

#### Global Variable Names

`all-lowercase`,`words_separated_by_underscore`

#### Function and variable Names

`lowercase`,`underscore_if_necessary`

#### Function/Method Arguments

Always use `self` for the first argument to instance methods.

Always use `cls` for the first argument to class methods.

If a function argument's name clashes with a reserved keyword, it is generally better to append a single trailing underscore rather than use an abbreviation or spelling corruption. Thus `class_` is better than `clss`. (Perhaps better is to avoid such clashes by using a synonym.)

#### Method Names and Instance Variables

`lowercase`,`underscore_if_necessary`

`_one_leading_underscore` only for non-public methods and instance variables.

#### Constants

`ALL_CAPITAL`