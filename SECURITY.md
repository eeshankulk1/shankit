# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue. Use GitHub's
private reporting: open the repository's **Security** tab and choose
**Report a vulnerability**.

Include what you found, the version or commit, and the steps to reproduce it.
We reply within a few working days, keep you posted while we fix it, and credit
you in the release notes unless you would rather stay anonymous.

## Supported versions

Throukit is pre-1.0. Fixes land on `main` and ship in the next release; older
releases do not get backports.

## Scope

Throukit runs the agent loop and passes tool calls to code you supply. Bugs in
that loop are in scope, for example a tool error that leaks internal details
to the model, or one run's context reaching another run. How your own tools
handle credentials, and what your connected services allow, are up to your
application.
