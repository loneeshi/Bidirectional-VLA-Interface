# Security

Do not commit provider keys, laboratory credentials, SSH configuration, bridge
spool contents, authorization identifiers, or machine-specific runtime data.
The bridge reads only explicitly supported provider variables from the process
environment; `.env` and local machine configuration are ignored by Git.

Report a suspected exposure privately to the repository maintainers. Revoke or
rotate the credential before discussing history cleanup. Git history is not
rewritten for usernames, local paths, or other non-secret provenance alone.
