# Security Policy

## Scope

PertEMA is a research tool that scores the reliability of a predictor's outputs. It runs locally or as a
self-hosted service. It does not require credentials and does not transmit user data to any third party. The
main security surfaces are the optional FastAPI service (input handling on the scoring endpoints) and the
dependencies.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Instead report it privately through GitHub's
"Report a vulnerability" flow on the repository's Security tab, or contact the maintainer through their GitHub
profile (https://github.com/OfficialBishal). Include a description, steps to reproduce, and the affected
version. You can expect an initial response within a reasonable time, and coordinated disclosure once a fix is
available.

## Supported versions

The project is pre-1.0 and moves quickly. Security fixes are applied to the latest release and to `main`.
