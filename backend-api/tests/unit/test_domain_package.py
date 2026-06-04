"""Tests for the domain package root."""

from __future__ import annotations


def test_domain_package_imports() -> None:
    import domain

    assert domain.__doc__

