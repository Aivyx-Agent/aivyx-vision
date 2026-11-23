# Changelog

## aivyx-vision-mold (unreleased version, main)

- Pins aivyx-route 0.3.0 (the "shared" tier); builds `LocalOnlyDns`
  with `LocalOnlyDns::new()`. No behaviour change.

## aivyx-vision-mold 0.1.1

- Local-only: the mold and broker HTTP client now uses aivyx-route's
  `LocalOnlyDns`, so a name that passes by its spelling but resolves to a
  public address is refused at connect time. Every request to mold or the
  broker is also checked with `require_local` first (reqwest skips the
  resolver for IP-literal URLs). A refusal is a `BackendUnreachable` error
  carrying the "... is not local (...)" text.
