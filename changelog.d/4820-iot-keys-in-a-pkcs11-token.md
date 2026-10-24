### Added: an IoT device key can live in a PKCS#11 token, never a PEM on disk

An `iot://` link whose `key=` is an RFC 7512 `pkcs11:` URI connects with the private key held in a hardware
security module (or SoftHSM) through `mtls_with_pkcs11`, so the key never enters the process or a file;
`module-path=` (or `PKCS11_MODULE`) names the driver and `pin-value=` (or `PKCS11_PIN`) the PIN. For provisioning
without a key file, `provision.register(..., csr_pem=, key=)` registers a certificate from an HSM-generated CSR and
writes only `cert.pem`, with the token URI in the returned query. A PEM key on disk stays the default.
