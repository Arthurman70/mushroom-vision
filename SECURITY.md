# Security

Keep the station bound to localhost unless you have configured its access token,
trusted HTTPS and a private network. Device workers receive restricted device
tokens rather than the station administrator token. Revoke a device when it is
lost or no longer trusted. Do not expose flight interfaces through this project.

Report a suspected vulnerability privately through the repository's security
advisory mechanism when available. Do not attach credentials, precise field
locations, contributor exports or private photographs to public issues.

This is an early research preview with no production security or aircraft
certification. Model inputs, imported labels and worker results remain untrusted.
