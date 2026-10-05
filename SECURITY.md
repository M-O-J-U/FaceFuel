# Security Policy

FaceFuel is a research prototype that processes photos of people's faces, eyes and
tongues. Please report anything that could expose that data or compromise the server.

## Reporting a vulnerability

**Do not open a public issue for security problems.** Email
**mojuaries111@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- the commit or version (`GET /health` reports it) you tested.

You should get an acknowledgement within 7 days. Please allow time for a fix before
disclosing publicly.

## Scope

In scope: `server.py`, the `facefuel/` package and the web frontend — e.g. upload
handling, image decoding, path handling, denial of service through crafted images,
or anything that could cause uploaded photos to be stored or leaked.

Out of scope: the medical validity of results (FaceFuel is not a medical device —
see the README), and third-party dependencies (report those upstream).

## Data handling

Uploaded images are processed in memory and are not written to disk or logged by
FaceFuel. If you find a code path where that is not true, please report it.
